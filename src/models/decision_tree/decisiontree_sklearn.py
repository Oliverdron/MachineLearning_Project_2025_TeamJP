import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

@dataclass
class DecisionTreeSklearnConfig:
    """
    Configuration for sklearn DecisionTreeRegressor.

    Args:
        criterion (str): Split quality metric ("squared_error", "friedman_mse")
        max_depth (Optional[int]): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf
        max_features (Optional[str]): Number of features considered at each split
        random_state (Optional[int]): Random seed
    """
    criterion: str = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[str] = None
    random_state: Optional[int] = None


class SklearnDecisionTreeRegressor:
    def __init__(self, cfg: DecisionTreeSklearnConfig):
        self.cfg = cfg
        self.model = DecisionTreeRegressor(cfg)
        
        self.criterion = cfg.criterion
        self.max_depth = cfg.max_depth
        self.min_samples_split = cfg.min_samples_split
        self.min_samples_leaf = cfg.min_samples_leaf
        self.max_features = cfg.max_features
        self.random_state = cfg.random_state
        
        if self.criterion not in {"squared_error", "friedman_mse"}:
            raise ValueError(f"Invalid criterion: {self.criterion}")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str) -> str:
        """
        Save the sklearn DecisionTreeRegressor to a compressed .npz file.

        Args:
            path (str): Path to save the model

        Returns:
            str: Path where the model was saved
        """
        np.savez_compressed(
            path,
            model=np.array(self.model, dtype=object),
            cfg=self.cfg,
        )

        logger.info(f"SklearnDecisionTreeRegressor saved to {path}")
        return path

# ---------------------------------------------------
# Decision Tree Regressor (Library Implementation)
# ---------------------------------------------------
# This model learns a set of hierarchical decision rules
# that split the feature space into regions.
# In each region (leaf), the prediction is the mean
# of the target values of samples in that region.
# ---------------------------------------------------

dt = DecisionTreeRegressor(
    # ---------------------------------------------
    # criterion
    # ---------------------------------------------
    # Defines how the quality of a split is measured.
    #
    # "squared_error":
    #   - minimizes Mean Squared Error (MSE)
    #   - equivalent to variance reduction
    #   - standard choice for regression trees
    #
    # Other options:
    #   - "absolute_error"  -> minimizes MAE
    #   - "friedman_mse"    -> used in gradient boosting
    #
    criterion="squared_error",

    # ---------------------------------------------
    # max_depth
    # ---------------------------------------------
    # Maximum depth of the tree.
    #
    # Depth = number of splits from root to leaf.
    # Controls model complexity:
    #   - small depth -> high bias, low variance
    #   - large depth -> low bias, high variance
    #
    # None means the tree grows until other stopping
    # criteria are met.
    #
    max_depth=5,

    # ---------------------------------------------
    # min_samples_split
    # ---------------------------------------------
    # Minimum number of samples required to split
    # an internal node.
    #
    # If a node has fewer samples than this value,
    # it becomes a leaf.
    #
    # Increasing this:
    #   - prevents deep trees
    #   - reduces overfitting
    #
    min_samples_split=10,

    # ---------------------------------------------
    # min_samples_leaf
    # ---------------------------------------------
    # Minimum number of samples required to be
    # present in EACH leaf node.
    #
    # A split is only allowed if both children
    # contain at least this many samples.
    #
    # This smooths predictions and reduces variance.
    #
    min_samples_leaf=5,

    # ---------------------------------------------
    # min_impurity_decrease
    # ---------------------------------------------
    # Minimum reduction in impurity (MSE) required
    # to perform a split.
    #
    # If the best possible split does not reduce
    # impurity by at least this amount, the node
    # becomes a leaf.
    #
    # Acts as a regularization parameter.
    #
    min_impurity_decrease=0.0,

    # ---------------------------------------------
    # random_state
    # ---------------------------------------------
    # Controls randomness for reproducibility.
    #
    # Important when:
    #   - multiple splits are equally good
    #   - used inside ensemble methods
    #
    random_state=42
)
#-------------------------------
#usage of the DecisionTreeRegressor
#-------------------------------

# Fit the decision tree to the training data
# X: shape (n_samples, n_features)
# y: shape (n_samples,)
# dt.fit(X, y)

# Predict target values for new samples
# y_pred = dt.predict(X_test)

# Depth of the trained tree
# tree_depth = dt.get_depth()

# Number of leaf nodes
# num_leaves = dt.get_n_leaves()

# Feature importance scores
# feature_importances = dt.feature_importances_