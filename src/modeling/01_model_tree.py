import numpy as np

class Node:
    """A node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorFromScratch:
    """
    A Decision Tree Regressor built from scratch (NumPy only).
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _mse(self, y):
        """Mean Squared Error for a node."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        """Find the best split according to MSE reduction."""
        n_samples, n_features = X.shape
        parent_mse = self._mse(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for t in thresholds:
                left_mask = X_col <= t
                right_mask = X_col > t

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                n_left = len(y_left)
                n_right = len(y_right)

                mse_left = self._mse(y_left)
                mse_right = self._mse(y_right)

                weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples
                gain = parent_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = t

        # Enforce minimum impurity decrease
        if best_gain < self.min_impurity_decrease:
            return None, None

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=np.mean(y))

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

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
        