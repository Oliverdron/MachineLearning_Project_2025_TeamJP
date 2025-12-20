import logging
import numpy as np

from sklearn.tree import DecisionTreeRegressor

logger = logging.get_logger(__name__)

class SklearnDecisionTreeRegressor:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.model = DecisionTreeRegressor(**cfg)

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
dt.fit(X, y)

# Predict target values for new samples
y_pred = dt.predict(X_test)

# Depth of the trained tree
tree_depth = dt.get_depth()

# Number of leaf nodes
num_leaves = dt.get_n_leaves()

# Feature importance scores
feature_importances = dt.feature_importances_

















#older version (left here for reference in case of need)

# import numpy as np
# class Node:
#     """A node in the decision tree."""
#     def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value 

#     def is_leaf(self):
#         return self.value is not None


# class DecisionTreeRegressorFromScratch:
#     """
#     A simple Decision Tree Regressor built from scratch.
#     """
#     def __init__(self, max_depth=None, min_samples_split=2):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.root = None

#     def fit(self, X, y):
#         self.root = self._build_tree(X, y, depth=0)
#         return self

#     def _build_tree(self, X, y, depth):
#         n_samples, n_features = X.shape

#         # Stopping conditions
#         if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
#             # Leaf node: mean value for regression
#             leaf_value = np.mean(y)
#             return Node(value=leaf_value)

#         # Find best split
#         best_feature, best_threshold = self._best_split(X, y)

#         # If no split improves the tree
#         if best_feature is None:
#             leaf_value = np.mean(y)
#             return Node(value=leaf_value)

#         # Split data
#         left_indices = X[:, best_feature] <= best_threshold
#         right_indices = X[:, best_feature] > best_threshold

#         # Recursively build subtrees
#         left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
#         right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

#         return Node(feature=best_feature, threshold=best_threshold,
#                     left=left_child, right=right_child)
    
#     def _mse(self, y):
#         """Calculate Mean Squared Error."""
#         if len(y) == 0:
#             return 0
#         mean = np.mean(y)
#         return np.mean((y - mean) ** 2)    

#     def _best_split(self, X, y):
#         n_samples, n_features = X.shape
#         if n_samples <= 1:
#             return None, None

#         best_gain = -float('inf')  # We want to MAXIMIZE MSE reduction
#         best_feature, best_threshold = None, None
#         parent_mse = self._mse(y)  # Parent MSE

#         # Try all features and thresholds
#         for feature_idx in range(n_features):
#             X_col = X[:, feature_idx]
#             thresholds = np.unique(X_col)
#             for t in thresholds:
#                 left_mask = X_col <= t
#                 right_mask = X_col > t

#                 if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
#                     continue

#                 # Calculate MSE for left and right splits
#                 n_left, n_right = len(y[left_mask]), len(y[right_mask])
#                 mse_left = self._mse(y[left_mask])
#                 mse_right = self._mse(y[right_mask])
                
#                 # Weighted MSE of children
#                 weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples
                
#                 # MSE reduction (gain)
#                 gain = parent_mse - weighted_mse

#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feature = feature_idx
#                     best_threshold = t

#         return best_feature, best_threshold

#     def predict(self, X):
#         return np.array([self._traverse_tree(x, self.root) for x in X])

#     def _traverse_tree(self, x, node):
#         if node.is_leaf():
#             return node.value
#         if x[node.feature] <= node.threshold:
#             return self._traverse_tree(x, node.left)
#         else:
#             return self._traverse_tree(x, node.right)