import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import compute_metrics
from src.models.results import ModelResult

logger = logging.getLogger(__name__)


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

    logger.info(f"Plotting residuals histogram to {out_path}")
    # Initialize plot
    plt.figure()
    # Plot residuals histogram
    plt.hist(resid, bins=50)
    # Set labels
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.tight_layout()
    # Save plot and close
    plt.savefig(out_path)
    plt.close()
    logger.info("Residuals histogram saved.")


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
        logger.info("All plots generated.")

    # Return the updated object so upstream code can chain
    return result
