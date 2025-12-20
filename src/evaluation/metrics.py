import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Target: regression tasks

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: RMSE value
    """
    # If one input comes in as a column vector while the other as a row vector, without reshaping subtraction is not possible
    # First convert inputs to numpy arrays and flatten to ensure element-wise operations work correctly
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    # Error per sample
    e = y_pred - y_true
    # Then take the average of squared errors and square root
    return float(np.sqrt(np.mean(e * e)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.

    Target: regression tasks

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: MAE value    
    """
    # First convert inputs to numpy arrays and flatten
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    # Error per sample
    e = y_pred - y_true
    # Then take the average of absolute errors
    return float(np.mean(np.abs(e)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R-squared between true and predicted values.
    
    Target: regression tasks

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: R-squared value
    """
    # First convert inputs to numpy arrays and flatten
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    # Sum of squared errors
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    # Total sum of squares
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    # Sanity check to avoid division by zero
    if ss_tot == 0:
        return float("nan")
    # Compute R-squared
    return 1.0 - ss_res / ss_tot


def poisson_nll(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute the Poisson Negative Log-Likelihood between true and predicted values.
    
    Target: claimNumber type data (non-negative integers)

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        eps (float): Small value to avoid log(0)

    Returns:
        float: Poisson NLL value
    """
    # First convert inputs to numpy arrays and flatten
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, None)
    # Poisson formula: -y_true * log(y_pred) + y_pred
    return float(np.mean(y_pred - y_true * np.log(y_pred)))


# Mapping of metric names to functions
METRICS = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "poisson_nll": poisson_nll,
}


def compute_metrics(metric_names: list, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.

    Args:
        metric_names (list): List of metric names to compute
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    # Define output dictionary
    out = {}
    # Compute each requested metric
    for m in metric_names:
        # Normalize metric name
        key = str(m).lower().strip()
        # Check if metric is known
        if key not in METRICS:
            raise ValueError(f"Unknown metric: {m}")
        # Compute and store metric
        logger.info(f"Computing metric: {key}")
        out[key] = METRICS[key](y_true, y_pred)
        logger.info(f"Metric {key}: {out[key]}")
    return out
