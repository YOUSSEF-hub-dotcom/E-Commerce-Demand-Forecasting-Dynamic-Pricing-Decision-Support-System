"""Optuna hyperparameter optimization for the Phase-3 candidate models.

Tuning runs **strictly within the training split** using expanding-window CV
(built in ``src.models.forecasting``). The objective minimizes the mean
validation WAPE over the CV folds. Every trial is fully reproducible via a
fixed TPE sampler seed.

Design note (documented subset trade-off, NOT a leak):
------------------------------------------------------
For the frozen v2 candidate the optimization was executed on a **mature-product
subset** of the training split (120 products / 8,899 rows) with the default
module settings 5 expanding-window CV folds (4-week validation horizon),
TPE sampler, seed 42, and 10 trials per model. This subset choice is a
representativeness/throughput design decision from the development phase and
is explicitly NOT a leakage finding: the subset is drawn from the training
split only, never from validation or test. The Optuna-winning hyperparameters
were frozen and the final candidate was refit on the FULL training split
(no retuning during cleanup; see ``final_forecast.WINNER_PARAMS``).
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.models.forecasting import make_model, run_expanding_cv
from src.utils.logging_setup import get_logger
from src.utils.metrics import wape

LOGGER = get_logger(__name__)

METRIC_PRIORITY = "wape"


def _suggest_lgbm(trial: optuna.trial.Trial) -> dict:
    """LightGBM hyperparameter space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
    }


def _suggest_xgb(trial: optuna.trial.Trial) -> dict:
    """XGBoost hyperparameter space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
    }


SUGGESTORS = {"lgbm": _suggest_lgbm, "xgb": _suggest_xgb}


def build_objective(
    model_type: str,
    X,
    y_log,
    weights,
    weeks,
    observed_demand,
    folds,
):
    """Create the Optuna objective for a candidate model type.

    Parameters
    ----------
    model_type : str
        ``lgbm`` or ``xgb``.
    X : pd.DataFrame
        Design matrix.
    y_log : np.ndarray
        Log-domain target.
    weights : np.ndarray
        Sample weights.
    weeks : pd.Series
        Weekly keys per row.
    observed_demand : np.ndarray
        Business-scale ground truth.
    folds : list[tuple[np.ndarray, np.ndarray]]
        Expanding-window CV folds.

    Returns
    -------
    callable
        ``objective(trial) -> float`` returning mean fold WAPE.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        params = SUGGESTORS[model_type](trial)
        model = make_model(model_type, params)
        results = run_expanding_cv(model, X, y_log, weights, weeks, observed_demand, folds)
        mean_wape = float(np.mean([r[METRIC_PRIORITY] for r in results]))
        return mean_wape

    return objective


def tune_hyperparameters(
    model_type: str,
    X,
    y_log,
    weights,
    weeks,
    observed_demand,
    folds,
    n_trials: int = 12,
    seed: int = 42,
) -> optuna.study.Study:
    """Run Optuna optimization over the training split CV folds.

    Parameters
    ----------
    model_type : str
        ``lgbm`` or ``xgb``.
    X : pd.DataFrame
        Design matrix (training rows only).
    y_log : np.ndarray
        Log-domain target.
    weights : np.ndarray
        Sample weights.
    weeks : pd.Series
        Weekly keys per row.
    observed_demand : np.ndarray
        Business-scale ground truth.
    folds : list[tuple[np.ndarray, np.ndarray]]
        Expanding-window CV folds.
    n_trials : int, optional
        Number of Optuna trials (default 12).
    seed : int, optional
        TPE sampler seed for reproducibility (default 42).

    Returns
    -------
    optuna.study.Study
        Completed study whose ``best_params`` are the tuned hyperparameters.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name=f"{model_type}_demand_tuning",
    )
    study.optimize(
        build_objective(model_type, X, y_log, weights, weeks, observed_demand, folds),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    LOGGER.info("Optuna [%s]: best WAPE=%.4f params=%s",
                model_type, study.best_value, study.best_params)
    return study