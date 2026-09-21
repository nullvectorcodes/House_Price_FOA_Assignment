"""
Visualization Module for House Price Prediction System.
Produces publication-grade plots using Matplotlib and Seaborn.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set clean modern style with Unicode-compatible font
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Arial Unicode MS", "Helvetica", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.autolayout": True,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.2,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    }
)


def plot_training_loss(
    history: Dict[str, List[float]],
    best_epoch: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots training and validation loss across epochs.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss (Cost)", color="#2563EB", lw=2)

    if val_loss:
        val_epochs = range(1, len(val_loss) + 1)
        ax.plot(val_epochs, val_loss, label="Validation Loss", color="#DC2626", lw=2, linestyle="--")

    if best_epoch and best_epoch <= len(val_loss):
        best_loss = val_loss[best_epoch - 1]
        ax.scatter(
            [best_epoch],
            [best_loss],
            color="#059669",
            s=100,
            zorder=5,
            label=f"Best Epoch ({best_epoch})",
        )
        ax.axvline(x=best_epoch, color="#059669", linestyle=":", alpha=0.7)

    ax.set_title("Training and Validation Cost Convergence", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax.set_ylabel("Loss / Cost", fontsize=11, fontweight="medium")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots actual vs predicted house prices with 45-degree reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Convert to Lakhs for readable axes
    y_true_lakhs = y_true / 100000.0
    y_pred_lakhs = y_pred / 100000.0

    ax.scatter(
        y_true_lakhs,
        y_pred_lakhs,
        alpha=0.45,
        color="#3B82F6",
        edgecolors="none",
        s=30,
        label="Test Properties",
    )

    min_val = min(float(np.min(y_true_lakhs)), float(np.min(y_pred_lakhs)))
    max_val = max(float(np.max(y_true_lakhs)), float(np.max(y_pred_lakhs)))
    line_coords = np.linspace(min_val, max_val, 100)
    ax.plot(line_coords, line_coords, color="#EF4444", lw=2, linestyle="--", label="Ideal Fit (y = x)")

    ax.set_title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Price (₹ Lakhs)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=11, fontweight="medium")
    ax.legend(frameon=True, facecolor="white")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots residuals vs predicted values and residual distribution.
    """
    residuals_lakhs = (y_true - y_pred) / 100000.0
    y_pred_lakhs = y_pred / 100000.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # Scatter of residuals vs predicted
    ax1.scatter(y_pred_lakhs, residuals_lakhs, alpha=0.45, color="#8B5CF6", s=25)
    ax1.axhline(0, color="#EF4444", linestyle="--", lw=2)
    ax1.set_title("Residuals vs. Predicted Value", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Predicted Price (₹ Lakhs)", fontsize=10)
    ax1.set_ylabel("Residual (Actual - Pred in ₹ Lakhs)", fontsize=10)

    # Histogram of residuals
    sns.histplot(residuals_lakhs, kde=True, ax=ax2, color="#06B6D4", bins=30)
    ax2.axvline(0, color="#EF4444", linestyle="--", lw=2)
    ax2.set_title("Residual Error Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Residual (₹ Lakhs)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance(
    feature_names: List[str],
    weights: np.ndarray,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots top standardized linear model coefficients.
    Note: Indicates standardized association magnitude, NOT causal effect.
    """
    abs_weights = np.abs(weights)
    indices = np.argsort(abs_weights)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_weights = weights[indices]
    colors = ["#10B981" if w > 0 else "#F43F5E" for w in top_weights]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    y_positions = range(len(top_features))

    bars = ax.barh(y_positions, top_weights, color=colors, alpha=0.85, edgecolor="#1E293B", lw=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="#64748B", linestyle="-", lw=1)

    ax.set_title(
        f"Top {top_n} Standardized Feature Coefficients\n(Green: Increases Price, Red: Decreases Price)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Standardized Weight Coefficient (w)", fontsize=10)

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        offset = 0.05 * (1 if width >= 0 else -1)
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9,
            color="#334155",
        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_price_distribution(
    y: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the distribution of house prices.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    y_lakhs = y / 100000.0

    sns.histplot(y_lakhs, kde=True, ax=ax, color="#6366F1", bins=35)
    mean_price = np.mean(y_lakhs)
    median_price = np.median(y_lakhs)

    ax.axvline(mean_price, color="#EF4444", linestyle="--", lw=2, label=f"Mean: ₹{mean_price:.1f}L")
    ax.axvline(median_price, color="#10B981", linestyle="-.", lw=2, label=f"Median: ₹{median_price:.1f}L")

    ax.set_title("Distribution of Residential Property Prices", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Price (₹ Lakhs)", fontsize=11)
    ax.set_ylabel("Number of Properties", fontsize=11)
    ax.legend(frameon=True, facecolor="white")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_area_vs_price(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of Area (sqft) vs Price, colored by location if available.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
    df_sample = df.sample(min(len(df), 1500), random_state=42)

    if "location" in df_sample:
        sns.scatterplot(
            data=df_sample,
            x="area_sqft",
            y=df_sample["price"] / 100000.0,
            hue="location",
            palette="Set2",
            alpha=0.6,
            ax=ax,
        )
    else:
        ax.scatter(df_sample["area_sqft"], df_sample["price"] / 100000.0, alpha=0.5, color="#0EA5E9")

    ax.set_title("Floor Area vs. Property Price", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Area (sq.ft)", fontsize=11)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=11)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
