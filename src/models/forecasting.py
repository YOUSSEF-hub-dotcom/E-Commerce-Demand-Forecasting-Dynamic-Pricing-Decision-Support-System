"""Forecasting models: factories, naive baseline, CV machinery, evaluation.

Implements the Phase-3 training workflow:

* ``make_model`` - normalized factory for LightGBM / XGBoost regressors.
* ``naive_combined_predictions`` - the no-skill benchmark (last observed demand
  per product, seeded from the training period).
* ``expanding_window_splits`` - expanding-window temporal CV folds. Folds are
  strictly chronological: each validation block sits *after* its training
  block, so no fold ever trains on future data.
* ``run_expanding_cv`` - evaluates any fit/predict model across folds and
  returns per-fold WAPE/MAE/RMSE/Asymmetric-Loss summaries.
* ``fit_full_with_early_stopping`` - final-data training with an early-stopping
  set carved from the chronological tail of the training split only (never
  from validation/test - zero lookahead remains absolute).
"""

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.metrics import describe_metrics, mae, rmse, wape

LOGGER = get_logger(__name__)


def make_model(model_type: str, params: dict):
    """Instantiate a regression model from a shared parameter dict.

    Parameters
    ----------
    model_type : str
        ``"lgbm"`` or ``"xgb"``.
    params : dict
        Model hyperparameters. ``n_estimators`` is honored by both
        implementations.

    Returns
    -------
    estimator
        Fitted-like estimator exposing ``fit(X, y, sample_weight=...)`` and
        ``predict(X)``.

    Raises
    ------
    ValueError
        If ``model_type`` is not supported.
    """
    p = dict(params)
    n_estimators = int(p.pop("n_estimators", 200))

    if model_type == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=n_estimators, objective="regression", verbose=-1, **p)
    if model_type == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=n_estimators, objective="reg:squarederror", verbosity=0, **p)

    raise ValueError(f"make_model: unsupported model_type '{model_type}'")


def naive_combined_predictions(train, evalset, target: str = "order_demand"):
    """Next-period forecast = the last observed demand per product.

    The training frame seeds the benchmark: an evaluation row inherits the most
    recent observed demand strictly before it, ensuring a genuine rolling
    persistence forecast rather than an in-split average.

    Parameters
    ----------
    train : pd.DataFrame
        Training feature frame (columns ``product_id``, ``week``, ``target``).
    evalset : pd.DataFrame
        Evaluation feature frame (same schema).
    target : str, optional
        Demand column (default ``order_demand``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y_true, pred)`` aligned to the evaluation rows that have history.
    """
    cols = ["product_id", "week", target]
    combined = pd.concat(
        [train[cols], evalset[cols]], ignore_index=True
    ).sort_values(["product_id", "week"], kind="mergesort")

    combined["pred"] = combined.groupby("product_id", observed=True)[target].shift(1)

    eval_weeks = set(evalset["week"].unique())
    mask = combined["week"].isin(eval_weeks) & combined["pred"].notna()
    out = combined.loc[mask, [target, "pred"]]
    return out[target].to_numpy(dtype=float), out["pred"].to_numpy(dtype=float)


def expanding_window_splits(
    unique_weeks,
    n_splits: int = 5,
    min_train_weeks: int = 20,
    horizon: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window chronological CV folds over weekly keys.

    Parameters
    ----------
    unique_weeks : array-like
        Sorted (or sortable) weekly Period keys available for CV.
    n_splits : int, optional
        Maximum number of folds to produce (default 5).
    min_train_weeks : int, optional
        Minimum training window in weeks for the first fold (default 20).
    horizon : int, optional
        Validation horizon in weeks per fold (default 4).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Fold list of ``(training_weeks, validation_weeks)`` arrays, built from
        the oldest data out. Stops before the window runs past the data.
    """
    weeks = np.asarray(sorted(unique_weeks))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        tr_end = min_train_weeks + i * horizon
        if tr_end + horizon > len(weeks):
            break
        folds.append((weeks[:tr_end], weeks[tr_end : tr_end + horizon]))
    LOGGER.info("expanding_window_splits: produced %d folds.", len(folds))
    return folds


def _fold_masks(weeks: pd.Series, train_weeks, val_weeks) -> tuple[np.ndarray, np.ndarray]:
    """Boolean index masks aligned to a frame for one fold."""
    tr = weeks.isin(train_weeks).to_numpy()
    va = weeks.isin(val_weeks).to_numpy()
    return tr, va


def run_expanding_cv(
    model,
    X,
    y_log,
    weights,
    weeks,
    observed_demand,
    folds,
) -> list[dict]:
    """Evaluate a fit/predict model across expanding-window folds.

    Each fold's predictions are exponentiated back to the business scale
    (``expm1``) and scored with the full metric suite.

    Parameters
    ----------
    model : estimator
        Any object with ``fit(X, y, sample_weight=...)`` and ``predict(X)``.
    X : pd.DataFrame
        Design matrix (rows aligned with ``weeks``/``observed_demand``).
    y_log : np.ndarray
        Log-domain target (``log1p(order_demand)``).
    weights : np.ndarray
        Sample weights aligned to rows.
    weeks : pd.Series
        Weekly Period key per row.
    observed_demand : np.ndarray
        Business-scale demand per row (ground truth).
    folds : list[tuple[np.ndarray, np.ndarray]]
        Output of ``expanding_window_splits``.

    Returns
    -------
    list[dict]
        Per-fold metric dictionaries: ``wape``, ``mae``, ``rmse``,
        ``asymmetric_loss_2_1``, plus ``n_train``/``n_val``.
    """
    results: list[dict] = []
    for idx, (tr_w, va_w) in enumerate(folds):
        tr_mask, va_mask = _fold_masks(weeks, tr_w, va_w)
        model.fit(X[tr_mask], y_log[tr_mask], sample_weight=weights[tr_mask])
        pred_log = model.predict(X[va_mask])
        pred = np.expm1(pred_log)
        y_true = observed_demand[va_mask]

        m = describe_metrics(y_true, pred)
        m["n_train"] = int(tr_mask.sum())
        m["n_val"] = int(va_mask.sum())
        results.append(m)
        LOGGER.info("CV fold %d: WAPE=%.4f MAE=%.2f RMSE=%.2f | train=%d val=%d",
                    idx + 1, m["wape"], m["mae"], m["rmse"], m["n_train"], m["n_val"])
    return results


def fit_full_with_early_stopping(
    model,
    X,
    y_log,
    weights,
    weeks,
    eval_weeks,
    early_stopping_rounds: int = 50,
):
    """Fit a model on the full training data with internal early stopping.

    The early-stopping set is the last ``eval_weeks`` weeks *of training only*.
    This remains strictly inside the training boundary, so no validation or
    test information is ever seen during fitting.

    Parameters
    ----------
    model : estimator
        LightGBM / XGBoost regressor (must support ``eval_set`` /
        ``early_stopping_rounds`` semantics).
    X : pd.DataFrame
        Design matrix.
    y_log : np.ndarray
        Log-domain target.
    weights : np.ndarray
        Sample weights.
    weeks : pd.Series
        Weekly keys per row.
    eval_weeks : array-like
        Chronological tail weeks of the training split used for early stopping.
    early_stopping_rounds : int, optional
        Rounds without improvement before stopping (default 50).

    Returns
    -------
    estimator
        The fitted model (best iteration restored).
    """
    eval_mask = weeks.isin(set(eval_weeks)).to_numpy()
    X_tr = X[~eval_mask].reset_index(drop=True)
    y_tr = np.asarray(y_log[~eval_mask])
    w_tr = np.asarray(weights[~eval_mask])
    eval_set = [(X[eval_mask].reset_index(drop=True), np.asarray(y_log[eval_mask]))]

    is_lgbm = model.__class__.__module__.startswith("lightgbm")
    if is_lgbm:
        from lightgbm import early_stopping

        fit = model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=eval_set,
            eval_metric="rmse",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        params = dict(getattr(model, "get_params", lambda: {})())
        params["early_stopping_rounds"] = early_stopping_rounds
        model = type(model)(**params)
        fit = model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=eval_set)
    LOGGER.info("fit_full_with_early_stopping: stopped at best_iteration=%s",
                getattr(fit, "best_iteration_", getattr(fit, "best_iteration", "?")))
    return fit


def score_on(pred_log: np.ndarray, observed_demand: np.ndarray) -> dict:
    """Score log-domain predictions against business-scale demand.

    Parameters
    ----------
    pred_log : np.ndarray
        Model predictions on the log1p target.
    observed_demand : np.ndarray
        Ground-truth ``order_demand``.

    Returns
    -------
    dict
        WAPE / MAE / RMSE / asymmetric-loss scores.
    """
    return describe_metrics(observed_demand, np.expm1(pred_log))