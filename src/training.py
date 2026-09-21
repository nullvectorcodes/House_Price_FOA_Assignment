"""
Training, Hyperparameter Tuning, and Cross-Validation Module.
Orchestrates model selection, early stopping, and generalization assessment.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation import calculate_metrics, root_mean_squared_error
from src.linear_regression import RegularizedLinearRegression


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rates: Optional[List[float]] = None,
    lambdas: Optional[List[float]] = None,
    penalties: Optional[List[Optional[str]]] = None,
    n_epochs: int = 1500,
    patience: int = 100,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs grid search across learning rates, regularization penalties, and lambdas.
    Evaluates strictly on validation RMSE to prevent selecting overfitted models.

    Parameters
    ----------
    X_train, y_train : Training features and targets.
    X_val, y_val : Validation features and targets.
    learning_rates : List of learning rates.
    lambdas : List of lambda regularization values.
    penalties : List of penalties (None, 'l1', 'l2').

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        (best_hyperparameters_dict, results_dataframe)
    """
    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01, 0.05]
    if lambdas is None:
        lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    if penalties is None:
        penalties = [None, "l1", "l2"]

    results: List[Dict[str, Any]] = []
    best_val_rmse = float("inf")
    best_params: Dict[str, Any] = {}

    for penalty in penalties:
        # If penalty is None, lambda 0 is sufficient
        cur_lambdas = [0.0] if penalty is None else lambdas

        for lr in learning_rates:
            for lmbda in cur_lambdas:
                model = RegularizedLinearRegression(
                    learning_rate=lr,
                    n_epochs=n_epochs,
                    penalty=penalty,
                    lmbda=lmbda,
                    patience=patience,
                )

                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                y_pred_val = model.predict(X_val)
                val_rmse = root_mean_squared_error(y_val, y_pred_val)

                y_pred_train = model.predict(X_train)
                train_rmse = root_mean_squared_error(y_train, y_pred_train)

                record = {
                    "penalty": "None" if penalty is None else str(penalty).upper(),
                    "learning_rate": lr,
                    "lambda": lmbda,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "best_epoch": model.best_epoch,
                    "final_epoch": model.final_epoch,
                    "early_stopped": model.early_stopped,
                }
                results.append(record)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params = {
                        "learning_rate": lr,
                        "penalty": penalty,
                        "lmbda": lmbda,
                        "n_epochs": n_epochs,
                        "patience": patience,
                        "val_rmse": val_rmse,
                    }

    results_df = pd.DataFrame(results).sort_values("val_rmse", ascending=True).reset_index(drop=True)
    return best_params, results_df


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    learning_rate: float = 0.01,
    penalty: Optional[str] = "l2",
    lmbda: float = 0.01,
    n_epochs: int = 1500,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Performs K-Fold Cross Validation (5 folds) to reliably estimate generalization error.

    Parameters
    ----------
    X, y : Complete training dataset.
    k : int, default=5
    learning_rate, penalty, lmbda, n_epochs : Hyperparameters.
    random_state : int, default=42

    Returns
    -------
    Dict[str, float]
        Mean and standard deviation of RMSE, MAE, and R².
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    rmse_scores: List[float] = []
    mae_scores: List[float] = []
    r2_scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = RegularizedLinearRegression(
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            penalty=penalty,
            lmbda=lmbda,
            patience=80,
        )
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        y_va_pred = model.predict(X_va)

        metrics = calculate_metrics(y_va, y_va_pred)
        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])
        r2_scores.append(metrics["r2"])

    return {
        "mean_rmse": float(np.mean(rmse_scores)),
        "std_rmse": float(np.std(rmse_scores)),
        "mean_mae": float(np.mean(mae_scores)),
        "std_mae": float(np.std(mae_scores)),
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
    }
