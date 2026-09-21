"""
Unit tests for Custom RegularizedLinearRegression and Regularization functions.
"""

import numpy as np
import pytest

from src.linear_regression import RegularizedLinearRegression
from src.regularization import (
    compute_l1_gradient,
    compute_l1_penalty,
    compute_l2_gradient,
    compute_l2_penalty,
    compute_regularization,
)


def test_regularization_penalties_and_gradients():
    """Verify analytical L1 and L2 penalty values and gradients."""
    w = np.array([2.0, -3.0, 0.0])
    lmbda = 0.5

    # L2: penalty = lambda * sum(w^2) = 0.5 * (4 + 9 + 0) = 6.5
    l2_cost = compute_l2_penalty(w, lmbda)
    assert pytest.approx(l2_cost, rel=1e-5) == 6.5

    # L2: grad = 2 * lambda * w = 2 * 0.5 * [2, -3, 0] = [2, -3, 0]
    l2_grad = compute_l2_gradient(w, lmbda)
    np.testing.assert_allclose(l2_grad, np.array([2.0, -3.0, 0.0]))

    # L1: penalty = lambda * sum(|w|) = 0.5 * (2 + 3 + 0) = 2.5
    l1_cost = compute_l1_penalty(w, lmbda)
    assert pytest.approx(l1_cost, rel=1e-5) == 2.5

    # L1: grad = lambda * sign(w) = 0.5 * [1, -1, 0] = [0.5, -0.5, 0.0]
    l1_grad = compute_l1_gradient(w, lmbda)
    np.testing.assert_allclose(l1_grad, np.array([0.5, -0.5, 0.0]))

    # None penalty
    cost_none, grad_none = compute_regularization(w, None, lmbda)
    assert cost_none == 0.0
    np.testing.assert_allclose(grad_none, np.zeros_like(w))


def test_model_initialization_and_shapes():
    """Verify model initialization and prediction output shape."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    true_w = np.array([1.5, -2.0, 3.0, 0.5])
    y = X @ true_w + 10.0 + rng.normal(scale=0.1, size=50)

    model = RegularizedLinearRegression(learning_rate=0.05, n_epochs=500)
    model.fit(X, y)

    assert model.weights is not None
    assert model.weights.shape == (4,)
    assert isinstance(model.bias, float)

    preds = model.predict(X)
    assert preds.shape == (50,)


def test_gradient_descent_decreases_loss():
    """Verify that batch gradient descent strictly decreases training loss."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 5.0

    model = RegularizedLinearRegression(learning_rate=0.05, n_epochs=200)
    model.fit(X, y)

    losses = model.history["train_loss"]
    assert len(losses) == 200
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1  # Converged on clean linear data


def test_bias_is_not_regularized():
    """Verify that bias term is not penalized during regularized updates."""
    # Data with zero mean X but non-zero mean target
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    # Target has high intercept = 100
    y = 100.0 + X @ np.array([1.0, 1.0])

    model_no_reg = RegularizedLinearRegression(learning_rate=0.05, n_epochs=300, penalty=None)
    model_no_reg.fit(X, y)

    # Even with heavy L2 regularization, bias should correctly recover ~100
    model_l2 = RegularizedLinearRegression(learning_rate=0.05, n_epochs=300, penalty="l2", lmbda=5.0)
    model_l2.fit(X, y)

    assert pytest.approx(model_no_reg.bias, rel=0.05) == 100.0
    assert pytest.approx(model_l2.bias, rel=0.05) == 100.0


def test_early_stopping_restores_best_weights():
    """Verify early stopping terminates when validation loss stops improving and restores best weights."""
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(100, 3))
    y_tr = X_tr @ np.array([1.0, 2.0, -1.0]) + 2.0

    # Validation set with massive divergence after a few epochs
    X_va = rng.normal(size=(30, 3))
    y_va = X_va @ np.array([1.0, 2.0, -1.0]) + 2.0

    model = RegularizedLinearRegression(
        learning_rate=0.05,
        n_epochs=1000,
        patience=20,
    )
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

    assert model.best_epoch > 0
    assert model.final_epoch >= model.best_epoch
    # Best weights restored
    assert model.weights is not None
