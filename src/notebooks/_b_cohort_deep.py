import json
import sys

sys.path.insert(0, r"D:\ALL Projects\E-Commerce")

import joblib
import numpy as np
import pandas as pd

from src.models.final_forecast import cohort_flags, inverse_log1p
from src.models.data_prep import design_matrix_rows

CACHE = r"C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments"
OUT = r"D:\ALL Projects\E-Commerce\reports\experiments"

prepared = joblib.load(CACHE + r"\final_lifecycle_phase3-exp2-v1_prepared.joblib")
model, (pred_val, val_act), (pred_test, test_act) = joblib.load(
    CACHE + r"\final_lifecycle_phase3-exp2-v1_model_and_preds.joblib"
)
tr = prepared["train"]
va = prepared["val"]

val_X, _, _ = design_matrix_rows(va)
pred_v = inverse_log1p(model.predict(val_X))
pred_log = model.predict(val_X)  # raw log-space output

val_pids = va["product_id"].astype(str).to_numpy()
val_cohort = cohort_flags(tr, val_pids)

f = pd.DataFrame(
    {
        "product_id": val_pids,
        "week": va["week"].astype(str).to_numpy(),
        "actual": va["order_demand"].astype("float64").to_numpy(),
        "pred": pred_v,
        "pred_log": pred_log,
        "cohort": val_cohort,
    }
)

res = {}

# ---- 1. log-space vs unit-space bias decomposition on B ---------------------
b = f[f.cohort == "B-sustained"]
pos = b.actual > 0
bpos = b[pos].copy()
log_actual = np.log1p(bpos.actual)
log_bias_mean = float((bpos.pred_log - log_actual).mean())  # mean signed log err
res["B_logspace_mean_signed_error"] = round(log_bias_mean, 4)
res["B_logspace_share_rows_under"] = round(float((bpos.pred_log < log_actual).mean()), 4)
res["B_unit_share_rows_under"] = round(float((bpos.pred < bpos.actual).mean()), 4)

# expm1/jensen contribution: model log-pred vs mean demand
actual_mean = bpos.actual.mean()
res["B_actual_mean"] = round(float(actual_mean), 3)
res["B_pos_actual_mean"] = round(float(bpos.actual.mean()), 3)
res["B_pred_mean"] = round(float(b.pred.mean()), 3)
res["B_pos_pred_mean"] = round(float(bpos.pred.mean()), 3)

# In log space is model centered? Compare expm1(mean log pred) to mean unit pred
res["B_expm1_mean_log_pred"] = round(float(np.expm1(bpos.pred_log.mean())), 3)
res["B_mean_unit_pred"] = round(float(bpos.pred.mean()), 3)

# ---- 2. zero-demand weeks inside B (val): what does model do? --------------
z = b[b.actual == 0]
nz = b[b.actual > 0]
res["B_zero_rows"] = int(len(z))
res["B_zero_rows_pred_mean"] = round(float(z.pred.mean()), 4)
res["B_zero_rows_pred_gt0"] = int((z.pred > 0).sum())
res["B_positive_rows_pred_mean"] = round(float(nz.pred.mean()), 3)
res["B_zero_rows_lastweek"] = int(((b.actual == 0) & (b.week == "2018-05-21")).sum())
res["B_rows_lastweek"] = int((b.week == "2018-05-21").sum())

# ---- 3. product concentration of the signed gap ------------------------------
g = b.groupby("product_id").agg(actual=("actual", "sum"), pred=("pred", "sum"), rows=("actual", "size"))
g["gap"] = g.actual - g.pred
g = g.sort_values("actual", ascending=False)
res["B_products_total"] = int(len(g))
res["B_products_under"] = int((g.actual > g.pred).sum())
res["B_products_over"] = int((g.actual < g.pred).sum())
res["B_gap_top5_share"] = round(float(g.gap.abs().head(5).sum() / g.gap.abs().sum()), 4)
res["B_gap_share_from_under"] = round(float(g[g.gap > 0].gap.sum() / g.gap.abs().sum()), 4)
res["B_products_gap_gt1sd"] = int((g.gap.abs() > g.gap.abs().std()).sum())
res["B_top5_detail"] = {
    str(k): {"actual": round(float(r.actual), 1), "pred": round(float(r.pred), 1), "gap": round(float(r.gap), 1)}
    for k, r in g.head(5).iterrows()
}

# ---- 4. instability: does weekly WAPE track a couple of volatile products? ----
wk = b.groupby("week").agg(actual=("actual", "sum"), pred=("pred", "sum"))
wk["bias_pct"] = (wk.pred / wk.actual - 1) * 100
res["B_weekly_bias_std"] = round(float(wk.bias_pct.std()), 2)
res["B_weekly_bias_range"] = [round(float(wk.bias_pct.min()), 2), round(float(wk.bias_pct.max()), 2)]
res["B_share_of_total_demand_units"] = round(float(f[f.cohort == "B-sustained"].actual.sum() / f.actual.sum()), 4)

# ---- 5. B2 contrast (upper-mid cohort, similar bias but less volatile?) -------
b2 = f[f.cohort == "B2-mid"]
wk2 = b2.groupby("week").agg(actual=("actual", "sum"), pred=("pred", "sum"))
wk2["bias_pct"] = (wk2.pred / wk2.actual - 1) * 100
res["B2_products_total"] = int(b2.product_id.nunique())
res["B2_weekly_bias_std"] = round(float(wk2.bias_pct.std()), 2)
res["B2_weekly_bias_mean"] = round(float(wk2.bias_pct.mean()), 2)
res["B2_zero_rows"] = int((b2.actual == 0).sum())

json.dump(res, open(rf"{OUT}\07b_b_cohort_deep.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))