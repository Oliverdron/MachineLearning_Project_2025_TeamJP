import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import compute_metrics
from src.models.results import ModelResult

logger = logging.getLogger(__name__)

# Colors for plotting
COLOR_PRIMARY = "#234C6A"
COLOR_SECONDARY  = "#0C2B4E" 

def plot_loss_curves(history: Dict[str, list], out_path: str) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history (Dict[str, list]): Training history containing 'loss' and optionally 'val_loss'
        out_path (str): Path to save the loss curve plot

    Returns:
        None
    """
    # Sanity check
    if not history:
        return
    
    # Extract loss values
    loss = history.get("loss")
    val_loss = history.get("val_loss")
    # Sanity check
    if not loss:
        return

    logger.info(f"Plotting loss curve to {out_path}")
    # Initialize plot
    plt.figure()
    # Basic 2D plot where x is epoch and y is loss
    plt.plot(loss, label="train_loss")
    # If we can plot validation loss as well
    if val_loss and not all(np.isnan(val_loss)):
        plt.plot(val_loss, label="val_loss")
    # Set labels and legend
    plt.title("Loss Curve(s)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # Save plot and close
    plt.savefig(out_path)
    plt.close()
    logger.info("Loss curve saved.")


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """
    Plot predicted vs true values scatter plot displays where the model landed and where it should have been.
    Plot identity line for reference, perfection is along this line.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        out_path (str): Path to save the plot

    Returns:
        None
    """
    # Make sure inputs are in the same shape to correctly compare them
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    logger.info(f"Plotting predicted vs true values to {out_path}")
    # Initialize plot
    plt.figure()
    plt.scatter(y_true, y_pred, s=10) # small dots
    # Plot identity line
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    # Set labels
    plt.title("Predicted vs True Values")
    plt.text(
            0.05, 0.95, "Ideal would be: y_pred = y_true", transform=plt.gca().transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_SECONDARY, alpha=0.8)
    )
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    # Save plot and close
    plt.savefig(out_path)
    plt.close()
    logger.info("Predicted vs True plot saved.")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """
    Plot histogram of residuals (y_true - y_pred).

    Interpretation:
        - Centered around 0 is good
        - Spread indicates variance of errors
        - Shifted positive indicates underprediction
        - Shifted negative indicates overprediction
        - Long tails indicate outliers

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        out_path (str): Path to save the plot

    Returns:
        None
    """
    # Make sure inputs are in the same shape to correctly compare them
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    # Calculate residuals
    resid = y_true - y_pred

    mu = float(np.mean(resid))
    sigma = float(np.std(resid, ddof=1)) # ddof for sample stddev

    logger.info(f"Plotting residuals histogram to {out_path}")
    # Initialize plot
    plt.figure()
    ax = plt.gca()
    # Plot residuals histogram
    ax.hist(resid, bins=range(resid.min().astype(int), resid.max().astype(int) + 1), color=COLOR_PRIMARY, edgecolor=COLOR_SECONDARY, alpha=0.85)
    # Overlay mean line
    ax.axvline(mu, color=COLOR_SECONDARY, linewidth=2)
    
    # Set labels
    ax.set_title("Residuals Histogram (e = y_true - y_pred)")   
    ax.set_xlabel("Positive: underprediction | Negative: overprediction")
    ax.set_xlim(resid.min(), resid.max())
    ax.set_ylabel("Count (log scale)")
    ax.set_yscale("log")

    # Stats label
    ax.text(
        0.98, 0.98,
        f"μ = {mu:.2f}\nσ = {sigma:.2f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_SECONDARY, alpha=0.8)
    )

    plt.tight_layout()
    # Save plot and close
    plt.savefig(out_path)
    plt.close()
    logger.info("Residuals histogram saved.")


def plot_r2_comparison(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, model_name: str = "") -> None:
    """
    Plot R^2 for:
      - Model predictions
      - Baseline: always predict 0
      - Baseline: always predict mean(y)

    Interpretation:
      - R^2 ~ 0 means you're basically at the mean baseline
      - Negative R^2 means worse than predicting mean(y)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_zero = np.zeros_like(y_true, dtype=float)
    y_mean = np.full_like(y_true, float(np.mean(y_true)), dtype=float)

    r2_model = compute_metrics(["r2"], y_true, y_pred)["r2"]
    r2_zero = compute_metrics(["r2"], y_true, y_zero)["r2"]
    r2_mean = compute_metrics(["r2"], y_true, y_mean)["r2"]

    labels = ["Model", "Always 0", "Mean(y)"]
    vals = [r2_model, r2_zero, r2_mean]

    logger.info(f"Plotting R2 comparison to {out_path}")
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)

    # Reference line: R² = 0 is the mean baseline
    plt.axhline(0.0, linestyle="--", linewidth=2, label="R² = 0 (mean baseline)")

    # Value labels
    for i, v in enumerate(vals):
        va = "bottom" if v >= 0 else "top"
        plt.text(i, v, f"{v:.4f}", ha="center", va=va)

    title = "R² comparison"
    if model_name:
        title += f" — {model_name}"
    plt.title(title)

    plt.ylabel("R² (higher is better)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_and_plot(result: ModelResult, eval_cfg: dict, plots_dir: Optional[str] = None) -> ModelResult:
    """
    Evaluate model results by computing metrics and generating plots.

    Args:
        result (ModelResult): ModelResult object containing predictions and true labels
        eval_cfg (dict): Evaluation configuration dictionary
        plots_dir (Optional[str]): Directory to save plots. (if None, uses eval_cfg setting)

    Returns:
        ModelResult: Updated ModelResult with computed metrics and plot paths
    """
    # Fetch metric names from config (default to RMSE and MAE)
    metric_names = eval_cfg.get("metrics", ["rmse", "mae"])
    # Compute metrics
    logger.info(f"Computing metrics: {metric_names} for model {result.model_name}")
    # Mutate the result object to add metrics
    result.metrics = compute_metrics(metric_names, result.y_true, result.y_pred)
    logger.info(f"Received metrics: {result.metrics}")

    # If no plots_dir provided, use from config (or default)
    if plots_dir is None:
        logger.debug("No plots_dir provided, fetching from eval_cfg.")
        plots_dir = eval_cfg.get("plots_dir", "reports/figures/models")
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    # Store plots directory in result
    result.plots_dir = plots_dir

    # Generate and save plots if enabled (default to True)
    if eval_cfg.get("save_plots", True):
        logger.info(f"Generating plots for model {result.model_name} in {plots_dir}")
        # Plot loss curves if training history is available
        if result.history:
            plot_loss_curves(result.history, os.path.join(plots_dir, f"{result.model_name}_loss.png"))
        plot_pred_vs_true(result.y_true, result.y_pred, os.path.join(plots_dir, f"{result.model_name}_pred_vs_true.png"))
        plot_residuals(result.y_true, result.y_pred, os.path.join(plots_dir, f"{result.model_name}_residuals.png"))
        plot_r2_comparison(result.y_true, result.y_pred, os.path.join(plots_dir, f"{result.model_name}_r2_comparison.png"), model_name=result.model_name)
        logger.info("All plots generated.")

    # Return the updated object so upstream code can chain
    return result
