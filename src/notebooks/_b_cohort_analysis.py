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

tr_pids = tr["product_id"].astype(str).to_numpy()
val_pids = va["product_id"].astype(str).to_numpy()
val_cohort = cohort_flags(tr, val_pids)

frame = pd.DataFrame(
    {
        "product_id": va["product_id"].astype(str).to_numpy(),
        "week": va["week"].astype(str),
        "actual": va["order_demand"].astype("float64").to_numpy(),
        "pred": pred_v,
        "cohort": val_cohort,
        "cat_share": va["category_share_prev"].astype("float64").to_numpy(),
        "price_prev": va["price_prev"].astype("float64").to_numpy(),
        "price_chg": va["price_chg_1w"].astype("float64").to_numpy(),
        "lag1": va["demand_lag_1"].astype("float64").to_numpy(),
        "lag2": va["demand_lag_2"].astype("float64").to_numpy(),
        "lag4": va["demand_lag_4"].astype("float64").to_numpy(),
        "rm8": va["rolling_demand_mean_8w"].astype("float64").to_numpy(),
        "rm16": va["rolling_demand_mean_16w"].astype("float64").to_numpy(),
        "rstd8": va["rolling_demand_std_8w"].astype("float64").to_numpy(),
        "ratio_4_12": va["regime_ratio_4_12"].astype("float64").to_numpy(),
        "ratio_8_16": va["regime_ratio_8_16"].astype("float64").to_numpy(),
        "pos_rate8": va["positive_rate_8w"].astype("float64").to_numpy(),
    }
)

res = {}
res["n_rows"] = {c: int((frame.cohort == c).sum()) for c in ("B-sustained", "B2-mid", "C-long-tail")}

# ---- 1. bias per week (under/over pattern) for B vs full ------------------
def weekly(frame_part):
    g = frame_part.groupby("week", sort=True)
    out = pd.DataFrame(
        {
            "actual": g.actual.sum(),
            "pred": g.pred.sum(),
            "wape": g.apply(lambda d: float(np.abs(d.actual - d.pred).sum() / d.actual.sum())),
            "bias_pct": (g.pred.sum() / g.actual.sum() - 1) * 100,
        }
    )
    return out

w_b = weekly(frame[frame.cohort == "B-sustained"])
w_all = weekly(frame)
res["B_weekly_bias"] = {str(k)[:10]: round(float(v), 3) for k, v in w_b.bias_pct.items()}
res["B_weekly_wape"] = {str(k)[:10]: round(float(v), 4) for k, v in w_b.wape.items()}
res["B_weeks_underforecast"] = int((w_b.bias_pct < 0).sum())
res["B_weeks_total"] = int(len(w_b))
res["all_weeks_underforecast"] = int((w_all.bias_pct < 0).sum())

# ---- 2. positivity / intermittency inside B (val) --------------------------
res["B_val_positive_rate"] = round(float(frame[frame.cohort == "B-sustained"].actual.gt(0).mean()), 4)
res["B_val_zero_rows"] = int((frame[frame.cohort == "B-sustained"].actual == 0).sum())
res["B_val_rows_Bpos"] = int((frame[frame.cohort == "B-sustained"].actual > 0).sum())

# recent acceleration regime (ratio_4_12 > 1.2 => recent growth)
b = frame[frame.cohort == "B-sustained"]
res["B_share_recent_growth_gt1_2"] = round(float((b.ratio_4_12 > 1.2).mean()), 4)
res["B_share_recent_decline_lt0_8"] = round(float((b.ratio_4_12 < 0.8).mean()), 4)
res["C_share_recent_growth_gt1_2"] = round(float((frame[frame.cohort == "C-long-tail"].ratio_4_12 > 1.2).mean()), 4)

# ---- 3. product-level concentration of B error -----------------------------
bg = frame[frame.cohort == "B-sustained"].groupby("product_id")
bprod = pd.DataFrame(
    {
        "actual": bg.actual.sum(),
        "pred": bg.pred.sum(),
        "wape": bg.apply(lambda d: float(np.abs(d.actual - d.pred).sum() / d.actual.sum()) if d.actual.sum() > 0 else np.nan),
        "pos_weeks": bg.actual.apply(lambda s: int((s > 0).sum())),
    }
)
bprod["bias_pct"] = (bprod.pred / bprod.actual - 1) * 100
bprod = bprod.sort_values("actual", ascending=False)
res["B_n_products"] = int(len(bprod))
res["B_top10_actual_share"] = round(float(bprod.actual.iloc[:10].sum() / bprod.actual.sum()), 4)
res["B_top50_actual_share"] = round(float(bprod.actual.iloc[:50].sum() / bprod.actual.sum()), 4)
res["B_top10_bias_pct"] = [round(float(v), 2) for v in bprod.bias_pct.iloc[:10]]
res["B_bias_pct_distribution"] = {
    "p10": round(float(bprod.bias_pct.quantile(0.1)), 2),
    "median": round(float(bprod.bias_pct.median()), 2),
    "p90": round(float(bprod.bias_pct.quantile(0.9)), 2),
}

# ---- 4. concentration among the 20 worst-WAPE B products -------------------
worst = bprod.dropna(subset=["wape"]).sort_values("wape", ascending=False).head(20)
res["B_worst20_wape_stats"] = {
    "mean_wape": round(float(worst.wape.mean()), 4),
    "actual_share": round(float(worst.actual.sum() / bprod.actual.sum()), 4),
    "typical_pos_weeks": round(float(worst.pos_weeks.mean()), 2),
}

# ---- 5. category concentration in B (derive category from one-hot cols) ----
cat_cols = [c for c in va.columns if c.startswith("cat_")]
bcat = pd.DataFrame(
    {
        "cat": va.loc[b.index, cat_cols].idxmax(axis=1).str.replace("cat_", "", regex=False),
        "actual": b.actual.values,
        "pred": b.pred.values,
        "wape": (np.abs(b.actual.values - b.pred.values) / b.actual.values.sum())
        if b.actual.values.sum() > 0
        else np.nan,
    }
)
bcat_sum = bcat.groupby("cat").agg(actual=("actual", "sum"), pred=("pred", "sum"), rows=("cat", "size"))
bcat_sum["wape"] = (bcat_sum.pred - bcat_sum.actual).abs() / bcat_sum.actual
bcat_sum = bcat_sum.sort_values("actual", ascending=False)
res["B_top5_categories"] = {
    str(k): {
        "actual": round(float(r.actual), 1),
        "wape": round(float(r.wape), 4),
        "bias_pct": round(float((r.pred / r.actual - 1) * 100), 2) if r.actual > 0 else None,
    }
    for k, r in bcat_sum.head(5).iterrows()
}

# ---- 6. price effects in B -------------------------------------------------
b_hi_price = b[b.price_prev > b.price_prev.median()]
res["B_hi_price_rows"] = int(len(b_hi_price))
res["B_hi_price_pos_rate"] = round(float((b_hi_price.actual > 0).mean()), 4)
res["B_price_chg_positive_share"] = round(float((b.price_chg > 0).mean()), 4)
res["B_price_chg_range"] = [round(float(b.price_chg.min()), 3), round(float(b.price_chg.max()), 3)]

# ---- 7. rolling/lag magnitude vs C ----------------------------------------
for col in ("lag1", "lag2", "lag4", "rm8", "rm16", "pos_rate8"):
    res[f"B_{col}_median"] = round(float(b[col].median()), 4)
    res[f"C_{col}_median"] = round(float(frame[frame.cohort == "C-long-tail"][col].median()), 4)

# ---- 8. global bias decomposition: does B dominate the -X% shortfall? ------
g_all = frame.groupby("cohort").agg(actual=("actual", "sum"), pred=("pred", "sum"))
g_all["bias_pct"] = (g_all.pred / g_all.actual - 1) * 100
g_all["actual_share"] = g_all.actual / g_all.actual.sum()
res["cohort_bias_pct"] = {str(k): round(float(v), 3) for k, v in g_all.bias_pct.items()}
res["cohort_actual_share"] = {str(k): round(float(v), 4) for k, v in g_all.actual_share.items()}
res["B_actual_total"] = float(g_all.actual["B-sustained"])
res["B_units_gap"] = float((g_all.actual - g_all.pred)["B-sustained"])

json.dump(res, open(rf"{OUT}\07_b_cohort_rootcause.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))