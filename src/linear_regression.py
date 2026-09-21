"""
Linear Regression with Batch Gradient Descent and Regularization.
Implemented from scratch using NumPy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.regularization import compute_regularization


class RegularizedLinearRegression:
    """
    Linear Regression model trained using Batch Gradient Descent with support
    for L1 (Lasso) and L2 (Ridge) regularization and Early Stopping.

    Hypothesis:
        y_hat = X @ w + b

    Cost Function:
        J(w, b) = (1/n) * sum((y_hat_i - y_i)^2) + R(w)

    Gradients:
        dw = (2/n) * X.T @ (y_hat - y) + dR/dw
        db = (2/n) * sum(y_hat - y)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 2000,
        penalty: Optional[str] = None,
        lmbda: float = 0.0,
        patience: int = 100,
        tol: float = 1e-7,
        random_state: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size for gradient descent updates.
        n_epochs : int, default=2000
            Maximum number of gradient descent iterations.
        penalty : str, optional ('l1', 'l2', or None)
            Type of regularization penalty.
        lmbda : float, default=0.0
            Regularization strength (lambda >= 0).
        patience : int, default=100
            Number of consecutive epochs with no validation loss improvement
            before triggering early stopping.
        tol : float, default=1e-7
            Minimum improvement threshold in loss.
        random_state : int, optional, default=42
            Seed for reproducibility.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")
        if lmbda < 0:
            raise ValueError("lmbda must be non-negative.")
        if patience <= 0:
            raise ValueError("patience must be a positive integer.")

        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.penalty = penalty.lower() if penalty else None
        self.lmbda = float(lmbda)
        self.patience = int(patience)
        self.tol = float(tol)
        self.random_state = random_state

        # Model parameters
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

        # Optimization & early stopping state
        self.best_weights: Optional[np.ndarray] = None
        self.best_bias: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.final_epoch: int = 0
        self.early_stopped: bool = False

        # Loss history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _init_parameters(self, n_features: int) -> None:
        """Initializes weights to zeros and bias to zero."""
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def compute_cost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        bias: Optional[float] = None,
    ) -> float:
        """
        Computes the regularized mean squared error cost:
        J(w, b) = (1/n) * sum((Xw + b - y)^2) + R(w)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        weights : np.ndarray, optional
            Weights vector. Defaults to self.weights.
        bias : float, optional
            Bias term. Defaults to self.bias.

        Returns
        -------
        float
            Total regularized cost.
        """
        w = self.weights if weights is None else weights
        b = self.bias if bias is None else bias

        if w is None:
            raise ValueError("Model has not been fitted or weights not provided.")

        n_samples = X.shape[0]
        y_hat = np.dot(X, w) + b
        mse = np.mean(np.square(y_hat - y))

        reg_penalty, _ = compute_regularization(w, self.penalty, self.lmbda)
        return float(mse + reg_penalty)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Computes analytical gradients of the regularized cost with respect to w and b.

        dw = (2/n) * X.T @ (y_hat - y) + dR/dw
        db = (2/n) * sum(y_hat - y)

        Note: Bias term is NOT regularized.
        """
        n_samples = X.shape[0]
        y_hat = np.dot(X, weights) + bias
        errors = y_hat - y  # shape: (n_samples,)

        # MSE gradients
        dw = (2.0 / n_samples) * np.dot(X.T, errors)
        db = (2.0 / n_samples) * float(np.sum(errors))

        # Add regularization gradient for weights only
        _, reg_grad = compute_regularization(weights, self.penalty, self.lmbda)
        dw = dw + reg_grad

        return dw, db

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RegularizedLinearRegression":
        """
        Fits the regularized linear model using Batch Gradient Descent and Early Stopping.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
        y_train : np.ndarray of shape (n_train,)
        X_val : np.ndarray of shape (n_val, n_features), optional
        y_val : np.ndarray of shape (n_val,), optional

        Returns
        -------
        self : RegularizedLinearRegression
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64).ravel()

        n_samples, n_features = X_tr.shape
        self._init_parameters(n_features)

        self.history = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.early_stopped = False
        epochs_without_improvement = 0

        for epoch in range(1, self.n_epochs + 1):
            # Compute gradients
            dw, db = self._compute_gradients(X_tr, y_tr, self.weights, self.bias)

            # Gradient Descent parameter updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute current training cost
            train_cost = self.compute_cost(X_tr, y_tr, self.weights, self.bias)
            self.history["train_loss"].append(train_cost)

            # Validation tracking and early stopping
            if has_val:
                val_cost = self.compute_cost(X_v, y_v, self.weights, self.bias)
                self.history["val_loss"].append(val_cost)

                if val_cost < (self.best_val_loss - self.tol):
                    self.best_val_loss = val_cost
                    self.best_epoch = epoch
                    self.best_weights = np.copy(self.weights)
                    self.best_bias = float(self.bias)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    self.early_stopped = True
                    self.final_epoch = epoch
                    # Restore best parameters
                    self.weights = np.copy(self.best_weights)
                    self.bias = float(self.best_bias)
                    break
            else:
                self.best_epoch = epoch
                self.best_weights = np.copy(self.weights)
                self.best_bias = float(self.bias)

            self.final_epoch = epoch

        # If loop completed without early stopping trigger and val set exists
        if has_val and not self.early_stopped and self.best_weights is not None:
            self.weights = np.copy(self.best_weights)
            self.bias = float(self.best_bias)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the learned linear model: y_hat = X @ w + b.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.weights) + self.bias

    def get_params(self) -> Dict[str, object]:
        """Returns hyperparameters dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "penalty": self.penalty,
            "lmbda": self.lmbda,
            "patience": self.patience,
            "tol": self.tol,
            "random_state": self.random_state,
        }
