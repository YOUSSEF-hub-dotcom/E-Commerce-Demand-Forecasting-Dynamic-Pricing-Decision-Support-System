# PART 5 — SHAP EXPLAINABILITY REPORT

Date (UTC): 2026-09-24T20:38:17.483319+00:00

## Model

- **MLflow model URI:** `models:/demand_forecasting_final/2`
- **Version:** 2 (stage: Production, status: READY)
- **Run ID:** `5d729a8594dc488bbd18ba0c4e216f5a`
- **Model type:** lightgbm.sklearn.LGBMRegressor
- **Feature count:** 93
- **PyFunc relationship:** `DemandForecaster` wrapper validates the 93-col schema and serves `expm1(log-output)` units; SHAP explains the wrapped LightGBM estimator's log1p output.

## Dataset

- **Data source:** frozen 93-column design rows, Train/Validation history only
- **Reference period:** 2018-05-14 (strictly before `TEST_START_WEEK = 2018-05-21`)
- **Window rows:** 32729
- **Rows explained:** 1500
- **Background sample:** 400
- **Test split accessed:** FALSE

## SHAP configuration

- **Explainer:** shap.explainers._tree.TreeExplainer (TreeExplainer on the Production estimator)
- **Expected value (log1p):** 0.090195
- **Background sample:** deterministic 400 rows from the reference week
- **Explanation sample:** deterministic 1500 rows from the reference week (excluding background)
- **Random seed:** 20260924 (`numpy` legacy `RandomState`)
- **SHAP scale:** log1p — unit prediction = `expm1(base + sum(shap))`

### Faithfulness (additivity)

- LightGBM internal feature binning makes `model.predict` float-sensitive at split boundaries, so raw SHAP `base + sum(phi)` can drift from the model output (max abs 0.026971, mean abs 0.002631, log1p scale).
- Per-row base values are therefore additivity-corrected: `base_row = model_log_prediction - sum(phi)`, so the served unit prediction is reproduced **exactly** as `expm1(base_row + sum(phi))`. Per-feature SHAP attributions are unchanged.

## Global findings (mean |SHAP|, top 12)

| # | Feature | Group | Mean |SHAP| | Typical direction |
|---|---------|-------|------:|--------------------|
| 1 | `cat_unknown` | categorical | 0.03042 | higher feature values are associated with lower predicted demand (log-domain). |
| 2 | `category_share_prev` | driver | 0.00738 | higher feature values are associated with lower predicted demand (log-domain). |
| 3 | `rolling_demand_std_8w` | regime | 0.00400 | higher feature values are associated with higher predicted demand (log-domain). |
| 4 | `cat_ferramentas_jardim` | categorical | 0.00248 | higher feature values are associated with higher predicted demand (log-domain). |
| 5 | `price_prev` | price | 0.00157 | no consistent monotone association in the explained sample. |
| 6 | `cat_cama_mesa_banho` | categorical | 0.00148 | higher feature values are associated with higher predicted demand (log-domain). |
| 7 | `rolling_demand_mean_8w` | regime | 0.00147 | higher feature values are associated with higher predicted demand (log-domain). |
| 8 | `demand_lag_1` | lag | 0.00139 | higher feature values are associated with higher predicted demand (log-domain). |
| 9 | `cat_esporte_lazer` | categorical | 0.00118 | higher feature values are associated with higher predicted demand (log-domain). |
| 10 | `rolling_demand_mean_4w` | rolling | 0.00102 | higher feature values are associated with higher predicted demand (log-domain). |
| 11 | `rolling_demand_mean_12w` | regime | 0.00096 | higher feature values are associated with higher predicted demand (log-domain). |
| 12 | `cat_informatica_acessorios` | categorical | 0.00089 | higher feature values are associated with higher predicted demand (log-domain). |

### Group-level mean |SHAP|

| Group | Total mean |SHAP| | Feature count |
|-------|------------:|:------------:|
| categorical | 0.0417 | 73 |
| regime | 0.0076 | 7 |
| driver | 0.0074 | 1 |
| price | 0.0016 | 1 |
| lag | 0.0014 | 3 |
| rolling | 0.0013 | 2 |
| freight | 0.0005 | 1 |
| price_change | 0.0000 | 1 |
| calendar | 0.0000 | 4 |

## Price-related features

| Feature | Group | Mean |SHAP| | Typical direction |
|---------|-------|------:|--------------------|
| `price_prev` | price | 0.00157 | no consistent monotone association in the explained sample. |
| `freight_prev` | freight | 0.00050 | higher feature values are associated with lower predicted demand (log-domain). |
| `price_chg_1w` | price_change | 0.00002 | higher feature values are associated with lower predicted demand (log-domain). |

Interpretation: the model associates these price/freight features with predicted demand, but SHAP is a model-attribution tool, not a demand-response estimator. Causal price-response belongs to the Part-4 elasticity analysis.

## Regime / temporal features

| Feature | Group | Mean |SHAP| | Typical direction |
|---------|-------|------:|--------------------|
| `rolling_demand_std_8w` | regime | 0.00400 | higher feature values are associated with higher predicted demand (log-domain). |
| `rolling_demand_mean_8w` | regime | 0.00147 | higher feature values are associated with higher predicted demand (log-domain). |
| `demand_lag_1` | lag | 0.00139 | higher feature values are associated with higher predicted demand (log-domain). |
| `rolling_demand_mean_4w` | rolling | 0.00102 | higher feature values are associated with higher predicted demand (log-domain). |
| `rolling_demand_mean_12w` | regime | 0.00096 | higher feature values are associated with higher predicted demand (log-domain). |
| `regime_ratio_4_12` | regime | 0.00035 | higher feature values are associated with higher predicted demand (log-domain). |
| `rolling_demand_mean_16w` | regime | 0.00033 | higher feature values are associated with higher predicted demand (log-domain). |
| `positive_rate_8w` | regime | 0.00028 | higher feature values are associated with higher predicted demand (log-domain). |
| `rolling_demand_std_4w` | rolling | 0.00026 | higher feature values are associated with higher predicted demand (log-domain). |
| `regime_ratio_8_16` | regime | 0.00019 | higher feature values are associated with lower predicted demand (log-domain). |
| `demand_lag_2` | lag | 0.00003 | higher feature values are associated with higher predicted demand (log-domain). |
| `demand_lag_4` | lag | 0.00002 | higher feature values are associated with lower predicted demand (log-domain). |

## Local explanations

| Example | Product | Week | Predicted demand (units) | Waterfall |
|---------|---------|------|--------------------------:|-----------|
| typical_demand | `50fd2b788dc166edd20512370dac54df` | `2018-05-14/2018-05-20` | 0.0239 | `D:\ALL Projects\E-Commerce\artifacts\explainability\local\example_typical_demand.png` |
| high_demand | `891fda5230e6e284e5430ded76eebcd7` | `2018-05-14/2018-05-20` | 0.0910 | `D:\ALL Projects\E-Commerce\artifacts\explainability\local\example_high_demand.png` |
| low_demand | `a8e69cb7ca2c04d5ba7f73b2385f2a76` | `2018-05-14/2018-05-20` | 0.0021 | `D:\ALL Projects\E-Commerce\artifacts\explainability\local\example_low_demand.png` |

## Limitations

- SHAP explains the model's behavior; it does not establish causality.
- Elasticity remains the appropriate framework for price-response analysis.
- SHAP importance does not imply business causality.
- Results depend on the reference data distribution (single validation week).
- Correlated features can distribute SHAP attribution across multiple variables.
