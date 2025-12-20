import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class RandomForestSklearnConfig:
    """
    Configuration for sklearn RandomForestRegressor.

    Args:
        n_estimators (int): Number of trees in the forest
        criterion (str): Split quality metric
        max_depth (Optional[int]): Maximum depth of each tree
        min_samples_split (int): Minimum samples required to split
        min_samples_leaf (int): Minimum samples per leaf
        max_features (str): Number of features considered at each split
        bootstrap (bool): Whether bootstrap samples are used
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (Optional[int]): Random seed
    """
    n_estimators: int = 100
    criterion: str = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: Optional[int] = None


class SklearnRandomForestRegressor:
    """
    Wrapper around sklearn's RandomForestRegressor
    with a unified save/load interface.
    """

    def __init__(self, cfg: RandomForestSklearnConfig):
        """
        Args:
            cfg (dict): Hyperparameters for RandomForestRegressor
        """
        
        self.cfg = cfg
        self.model = RandomForestRegressor(cfg)
        self.n_estimators = cfg.n_estimators
        self.criterion = cfg.criterion
        self.max_depth = cfg.max_depth
        self.min_samples_split = cfg.min_samples_split
        self.min_samples_leaf = cfg.min_samples_leaf
        self.max_features = cfg.max_features
        self.bootstrap = cfg.bootstrap
        self.n_jobs = cfg.n_jobs
        self.random_state = cfg.random_state
        
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
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
        Save the Random Forest model to a compressed .npz file.

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

        logger.info(f"SklearnRandomForestRegressor saved to {path}")
        return path



# ---------------------------------------------------
# Random Forest Regressor (Library Implementation)
# ---------------------------------------------------
# A Random Forest is an ENSEMBLE of decision trees.
#
# Key idea:
# - Train many decision trees independently
# - Each tree sees a different bootstrap sample of the data
# - Each split considers only a random subset of features
# - Final prediction = average of all tree predictions
#
# This reduces variance compared to a single decision tree
# while keeping low bias.
# ---------------------------------------------------

rf = RandomForestRegressor(
    # ---------------------------------------------
    # n_estimators
    # ---------------------------------------------
    # Number of decision trees in the forest.
    #
    # More trees:
    #   - lower variance
    #   - more stable predictions
    #   - higher computational cost
    #
    # Typical values: 100–500
    #
    n_estimators=200,

    # ---------------------------------------------
    # criterion
    # ---------------------------------------------
    # Metric used to measure split quality inside
    # each individual decision tree.
    #
    # "squared_error":
    #   - minimizes Mean Squared Error (MSE)
    #   - default for regression forests
    #
    criterion="squared_error",

    # ---------------------------------------------
    # max_depth
    # ---------------------------------------------
    # Maximum depth of each individual tree.
    #
    # Limiting depth:
    #   - reduces overfitting
    #   - increases bias slightly
    #
    # If None, trees grow until other stopping
    # criteria are met.
    #
    max_depth=None,

    # ---------------------------------------------
    # min_samples_split
    # ---------------------------------------------
    # Minimum number of samples required to split
    # an internal node in EACH tree.
    #
    min_samples_split=2,

    # ---------------------------------------------
    # min_samples_leaf
    # ---------------------------------------------
    # Minimum number of samples required in each
    # leaf node.
    #
    # Helps smooth predictions and reduce variance.
    #
    min_samples_leaf=1,

    # ---------------------------------------------
    # max_features
    # ---------------------------------------------
    # Number of features considered when looking
    # for the best split.
    #
    # This is CRUCIAL for Random Forests.
    #
    # Common choices:
    #   - "sqrt"  -> sqrt(n_features) (default)
    #   - "log2"  -> log2(n_features)
    #   - None    -> all features (NOT recommended)
    #
    # Random feature selection decorrelates trees.
    #
    max_features="sqrt",

    # ---------------------------------------------
    # bootstrap
    # ---------------------------------------------
    # Whether bootstrap samples are used when
    # building trees.
    #
    # True:
    #   - each tree sees a different random sample
    #   - sampling is done with replacement
    #
    bootstrap=True,

    # ---------------------------------------------
    # oob_score
    # ---------------------------------------------
    # Whether to use Out-Of-Bag (OOB) samples
    # to estimate generalization error.
    #
    # Each tree leaves out ~36% of samples.
    # These can be used as a validation set.
    #
    oob_score=True,

    # ---------------------------------------------
    # n_jobs
    # ---------------------------------------------
    # Number of CPU cores used for training.
    #
    # -1 means use ALL available cores.
    #
    n_jobs=-1,

    # ---------------------------------------------
    # random_state
    # ---------------------------------------------
    # Controls randomness for reproducibility.
    #
    random_state=42
)

# ---------------------------------------------
#Usage of the Random Forest Regressor
# ---------------------------------------------

# Fit the random forest to the training data
# rf.fit(X, y)

# Predict by averaging predictions from all trees
# y_pred = rf.predict(X_test)

# Number of trees
# num_trees = len(rf.estimators_)

# Out-of-bag R^2 score (if oob_score=True)
# oob_r2 = rf.oob_score_

# Feature importance (averaged over all trees)
# feature_importances = rf.feature_importances_
