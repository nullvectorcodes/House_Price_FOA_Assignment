"""
Evaluation and Metrics Module for House Price Prediction.

Calculates:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)
- Indian Rupee Currency Formatting (Lakhs / Crores)
- Prediction Error Confidence Intervals
"""

from typing import Dict, Tuple
import numpy as np


def format_inr(number: float) -> str:
    """
    Formats a numeric value into the Indian currency system (e.g., ₹92,50,000).

    Parameters
    ----------
    number : float or int

    Returns
    -------
    str
        Formatted string prefixed with '₹'.
    """
    if np.isnan(number):
        return "₹0"

    is_negative = number < 0
    num_abs = abs(round(number))
    num_str = str(num_abs)

    if len(num_str) <= 3:
        formatted = num_str
    else:
        last_three = num_str[-3:]
        remaining = num_str[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last_three

    return f"-₹{formatted}" if is_negative else f"₹{formatted}"


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAE = (1/n) * sum(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MSE = (1/n) * sum((y_true - y_pred)^2)."""
    return float(np.mean(np.square(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes RMSE = sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes R² = 1 - (SS_res / SS_tot).
    SS_res = sum((y_true - y_pred)^2)
    SS_tot = sum((y_true - mean(y_true))^2)
    """
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAPE in percentage (%)."""
    denominator = np.where(y_true == 0, 1e-5, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates comprehensive model evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    Dict[str, float]
        Dictionary with mae, mse, rmse, r2, mape.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    rmse = root_mean_squared_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def compute_confidence_band(
    predicted_price: float,
    model_rmse: float,
    confidence_multiplier: float = 1.96,
) -> Tuple[float, float]:
    """
    Computes a 95% approximate empirical error band for a prediction based on test RMSE.

    Parameters
    ----------
    predicted_price : float
    model_rmse : float
    confidence_multiplier : float, default=1.96 (~95% normal interval)

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    margin = confidence_multiplier * model_rmse
    lower = max(0.0, predicted_price - margin)
    upper = predicted_price + margin
    return lower, upper
