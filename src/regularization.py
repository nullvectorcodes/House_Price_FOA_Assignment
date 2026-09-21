"""
Regularization Module for Linear Regression.
Implements L1 (Lasso) and L2 (Ridge) penalty and gradient calculations.

Mathematical Notes:
- L2 penalty: R(w) = lambda * sum(w_j^2)
  L2 gradient: dR/dw = 2 * lambda * w
- L1 penalty: R(w) = lambda * sum(|w_j|)
  L1 gradient: dR/dw = lambda * sign(w)
- Bias term is NEVER regularized, as doing so would shift the baseline prediction toward zero regardless of the target mean.
"""

from typing import Tuple
import numpy as np


def compute_l2_penalty(weights: np.ndarray, lmbda: float) -> float:
    """
    Computes the L2 (Ridge) regularization penalty: lambda * sum(w_j^2).

    Parameters
    ----------
    weights : np.ndarray
        Model weights vector of shape (d,).
    lmbda : float
        Regularization strength hyperparameter (lambda >= 0).

    Returns
    -------
    float
        The scalar L2 penalty value.
    """
    if lmbda <= 0.0:
        return 0.0
    return float(lmbda * np.sum(np.square(weights)))


def compute_l2_gradient(weights: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Computes the gradient of the L2 penalty with respect to weights: 2 * lambda * w.

    Parameters
    ----------
    weights : np.ndarray
        Model weights vector of shape (d,).
    lmbda : float
        Regularization strength hyperparameter.

    Returns
    -------
    np.ndarray
        Regularization gradient vector of shape (d,).
    """
    if lmbda <= 0.0:
        return np.zeros_like(weights, dtype=np.float64)
    return 2.0 * lmbda * weights


def compute_l1_penalty(weights: np.ndarray, lmbda: float) -> float:
    """
    Computes the L1 (Lasso) regularization penalty: lambda * sum(|w_j|).

    Parameters
    ----------
    weights : np.ndarray
        Model weights vector of shape (d,).
    lmbda : float
        Regularization strength hyperparameter (lambda >= 0).

    Returns
    -------
    float
        The scalar L1 penalty value.
    """
    if lmbda <= 0.0:
        return 0.0
    return float(lmbda * np.sum(np.abs(weights)))


def compute_l1_gradient(weights: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Computes the subgradient of the L1 penalty with respect to weights: lambda * sign(w).

    Parameters
    ----------
    weights : np.ndarray
        Model weights vector of shape (d,).
    lmbda : float
        Regularization strength hyperparameter.

    Returns
    -------
    np.ndarray
        Subgradient vector of shape (d,).
    """
    if lmbda <= 0.0:
        return np.zeros_like(weights, dtype=np.float64)
    return lmbda * np.sign(weights)


def compute_regularization(
    weights: np.ndarray, penalty: str | None, lmbda: float
) -> Tuple[float, np.ndarray]:
    """
    Computes regularization penalty and gradient based on chosen penalty type.

    Parameters
    ----------
    weights : np.ndarray
        Model weights vector of shape (d,).
    penalty : str or None
        'l1', 'l2', or None / 'none'.
    lmbda : float
        Regularization strength parameter.

    Returns
    -------
    Tuple[float, np.ndarray]
        (penalty_cost, regularization_gradient)
    """
    if penalty is None or str(penalty).lower() in ("none", "") or lmbda <= 0.0:
        return 0.0, np.zeros_like(weights, dtype=np.float64)

    penalty_str = str(penalty).lower().strip()
    if penalty_str == "l2":
        cost = compute_l2_penalty(weights, lmbda)
        grad = compute_l2_gradient(weights, lmbda)
        return cost, grad
    elif penalty_str == "l1":
        cost = compute_l1_penalty(weights, lmbda)
        grad = compute_l1_gradient(weights, lmbda)
        return cost, grad
    else:
        raise ValueError(
            f"Unsupported penalty '{penalty}'. Supported values: 'l1', 'l2', None."
        )
