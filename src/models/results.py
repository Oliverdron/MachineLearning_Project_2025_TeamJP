from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

# Use dataclass to automatically generate init, repr, etc.
@dataclass
class ModelResult:
    """
    Data class to store results and metadata for a trained model.

    Attributes:
        model_name (str): Name of the model
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_train (Optional[np.ndarray]): Predicted labels for training data
        history (Dict[str, list]): Training history (e.g., loss, accuracy over epochs)
        metrics (Dict[str, float]): Evaluation metrics (e.g., accuracy, F1-score)
        params (Dict[str, Any]): Model hyperparameters
        model_artifact_path (Optional[str]): Path to saved model artifact
        plots_dir (Optional[str]): Directory containing related plots
    """
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray

    y_pred_train: Optional[np.ndarray] = None
    history: Dict[str, list] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    stage: str = "unknown"  # e.g.: "cv", "test"
    feature_set: str = "unknown"  # e.g.: "tree", "nn"

    model_artifact_path: Optional[str] = None
    plots_dir: Optional[str] = None
