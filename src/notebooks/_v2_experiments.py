import json
import os
import sys

sys.path.insert(0, r"D:\ALL Projects\E-Commerce")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.final_forecast import (
    cohort_flags,
    compute_mase_denominator,
    freeze_config,
    full_metric_suite,
    inverse_log1p,
)
from src.models.data_prep import design_matrix_rows

CACHE = r"C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments"
OUT = r"D:\ALL Projects\E-Commerce\reports\experiments"
RESULT_FILE = rf"{OUT}\07c_v2_experiments_val.json"

results = {}
if os.path.exists(RESULT_FILE):
    results = json.load(open(RESULT_FILE, encoding="utf-8"))
    print("loaded existing results keys:", list(results.keys()))

prepared = joblib.load(CACHE + r"\final_lifecycle_phase3-exp2-v1_prepared.joblib")
tr_full = prepared["train"]
va = prepared["val"]

train_X, train_y, train_w = design_matrix_rows(tr_full)
val_X, _, _ = design_matrix_rows(va)
val_actual = va["order_demand"].astype("float64").to_numpy()
mase_denom = compute_mase_denominator(tr_full)

val_pids = va["product_id"].astype(str).to_numpy()
tr_pids = tr_full["product_id"].astype(str).to_numpy()
val_cohort = cohort_flags(tr_full, val_pids)

tr_pid = tr_full["product_id"].astype(str).to_numpy()
train_cohort = cohort_flags(tr_full, tr_pid)


def suites_done():
    return "E3_weighted_calibrated" in results and "E5_tweedie_unitspace" in results


def fit(weights, x, y, objective=None, **extra):
    cfg = freeze_config()
    hp = {k: v for k, v in cfg["params"].items()}
    if objective is not None:
        hp = {"objective": objective, "metric": "tweedie" if objective == "tweedie" else "rmse", **hp}
    hp.update(extra)
    model = lgb.LGBMRegressor(**hp)
    model.fit(x, y, sample_weight=weights)
    return model


def suite(pred):
    return {k: round(float(v), 6) for k, v in full_metric_suite(val_actual, pred, val_cohort, mase_denom).items()}


def integrity(pred):
    return {"n_neg": int((pred < 0).sum()), "n_nan_inf": int((~np.isfinite(pred)).sum())}


def base_weights(k_b=1.0, k_b2=1.0):
    w = train_w.astype("float64").copy()
    b = train_cohort == "B-sustained"
    b2 = train_cohort == "B2-mid"
    w[b] *= k_b
    w[b2] *= k_b2
    return w


# ---- E0: frozen EXP2 reference (retrained, identical to v1 model) -----------
if "E0_frozen_reference" not in results:
    m0 = fit(train_w, train_X, train_y)
    p0 = inverse_log1p(m0.predict(val_X))
    results["E0_frozen_reference"] = suite(p0)
    results["E0_pred_integrity"] = integrity(p0)
    joblib.dump(m0, CACHE + r"\v2_exp_E0.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

# ---- E1: B/B2-aware sample weights ------------------------------------------
if "E1_BB2_weighted" not in results:
    w1 = base_weights(k_b=10.0, k_b2=5.0)
    m1 = fit(w1, train_X, train_y)
    p1 = inverse_log1p(m1.predict(val_X))
    results["E1_BB2_weighted"] = suite(p1)
    results["E1_pred_integrity"] = integrity(p1)
    results["E1_weight_stats"] = {
        "B_rows_avg_effective_mult": float(base_weights(k_b=10.0, k_b2=5.0)[train_cohort == "B-sustained"].mean()),
        "C_rows_avg_mult": round(float(train_w[train_cohort == "C-long-tail"].mean()), 4),
    }
    joblib.dump(m1, CACHE + r"\v2_exp_E1.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

# ---- E2: train-derived per-cohort level calibration --------------------------
weeks = sorted(tr_full["week"].astype(str).unique())
cut_week = weeks[-8]
tail_mask = tr_full["week"].astype(str) >= cut_week
head = tr_full[~tail_mask]
tx_h, ty_h, tw_h = design_matrix_rows(head)
tx_t, ty_t, tw_t = design_matrix_rows(tr_full[tail_mask])

if "E2_calibrated" not in results:
    m_head = fit(tw_h, tx_h, ty_h)
    tail_actual_all = tr_full.loc[tail_mask, "order_demand"].astype("float64").to_numpy()
    tail_pids = tr_full.loc[tail_mask, "product_id"].astype(str).to_numpy()
    tail_cohort = cohort_flags(head, tail_pids)
    tail_pred = inverse_log1p(m_head.predict(tx_t))


    def per_cohort_factor(pred, actual, cohort):
        fac = {}
        for label in ("B-sustained", "B2-mid", "C-long-tail"):
            m = cohort == label
            s = actual[m].sum()
            fac[label] = 1.0 if s <= 0 or pred[m].sum() <= 0 else float(s / pred[m].sum())
        return {k: float(np.clip(v, 0.7, 2.0)) for k, v in fac.items()}


    E2_FAC = per_cohort_factor(tail_pred, tail_actual_all, tail_cohort)
    m2 = fit(train_w, train_X, train_y)
    p2_raw = inverse_log1p(m2.predict(val_X))
    p2 = np.asarray([x * E2_FAC.get(lbl, 1.0) for x, lbl in zip(p2_raw, val_cohort)], dtype=float)
    results["E2_calibrated"] = suite(p2)
    results["E2_pred_integrity"] = integrity(p2)
    results["E2_factors"] = E2_FAC
    joblib.dump({"model": m2, "factor": E2_FAC, "labels": list(val_cohort)}, CACHE + r"\v2_exp_E2.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

# ---- E3: weighting + calibration combined -----------------------------------
if "E3_weighted_calibrated" not in results:
    m3 = fit(base_weights(k_b=10.0, k_b2=5.0), train_X, train_y)
    p3_raw = inverse_log1p(m3.predict(val_X))
    if "E2_factors" in results:
        E2_FAC_3 = results["E2_factors"]
    else:
        m_head = fit(tw_h, tx_h, ty_h)
        tail_actual_all = tr_full.loc[tail_mask, "order_demand"].astype("float64").to_numpy()
        tail_pids = tr_full.loc[tail_mask, "product_id"].astype(str).to_numpy()
        tail_cohort = cohort_flags(head, tail_pids)
        tail_pred = inverse_log1p(m_head.predict(tx_t))
        E2_FAC_3 = per_cohort_factor(tail_pred, tail_actual_all, tail_cohort)
    p3 = np.asarray([x * E2_FAC_3.get(lbl, 1.0) for x, lbl in zip(p3_raw, val_cohort)], dtype=float)
    results["E3_weighted_calibrated"] = suite(p3)
    results["E3_pred_integrity"] = integrity(p3)
    joblib.dump({"model": m3, "factor": E2_FAC_3, "labels": list(val_cohort)}, CACHE + r"\v2_exp_E3.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

# ---- E4: cohort-level Jensen variance correction (train-derived) ------------
if "E4_jensen_corrected" not in results:
    m_head = fit(tw_h, tx_h, ty_h)
    tail_log_pred = m_head.predict(tx_t)
    tail_log_actual = np.log1p(np.asarray(tr_full.loc[tail_mask, "order_demand"], dtype=float))
    tail_log_res = tail_log_pred - tail_log_actual
    tail_pids = tr_full.loc[tail_mask, "product_id"].astype(str).to_numpy()
    tail_cohort = cohort_flags(head, tail_pids)
    sigma2 = {}
    for label in ("B-sustained", "B2-mid", "C-long-tail"):
        m = tail_cohort == label
        sigma2[label] = float(np.var(tail_log_res[m])) if m.sum() > 0 else 0.0
    results["E4_sigma2"] = {k: round(v, 6) for k, v in sigma2.items()}
    m4 = fit(train_w, train_X, train_y)
    p4_log = m4.predict(val_X)
    p4 = inverse_log1p(p4_log + np.asarray([sigma2.get(lbl, 0.0) / 2.0 for lbl in val_cohort]))
    results["E4_jensen_corrected"] = suite(p4)
    results["E4_pred_integrity"] = integrity(p4)
    joblib.dump({"model": m4, "sigma2": sigma2, "labels": list(val_cohort)}, CACHE + r"\v2_exp_E4.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

# ---- E5: unit-space Tweedie objective ---------------------------------------
if "E5_tweedie_unitspace" not in results:
    m5 = fit(np.ones(len(tr_full)), train_X, np.asarray(tr_full["order_demand"], dtype=float).ravel(),
             objective="tweedie")
    p5_log = np.log1p(np.maximum(m5.predict(val_X), 0.0))
    p5 = inverse_log1p(p5_log)
    results["E5_tweedie_unitspace"] = suite(p5)
    results["E5_pred_integrity"] = integrity(p5)
    joblib.dump(m5, CACHE + r"\v2_exp_E5.joblib")
    json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)

json.dump(results, open(RESULT_FILE, "w"), indent=2, default=str)
print(json.dumps(results, indent=2, default=str))