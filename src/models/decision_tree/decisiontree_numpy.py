import numpy as np
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# Node class (tree building block)
# -------------------------------
class Node:
    """
    A single node in the decision tree.

    This node can be either:
    - an INTERNAL (decision) node: stores a feature index + threshold and has left/right children
    - a LEAF node: stores a prediction value (the mean target value of samples in that leaf)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # feature: integer index of the feature used to split at this node (e.g., 0, 1, 2, ...)
        self.feature = feature

        # threshold: numeric cutoff for the split
        # rule: if X[:, feature] <= threshold -> go left, else -> go right
        self.threshold = threshold

        # left child node (subtree for samples meeting the <= threshold condition)
        self.left = left

        # right child node (subtree for samples > threshold)
        self.right = right

        # value: if this node is a leaf, value holds the predicted output (a scalar)
        # if value is not None -> leaf node; if None -> internal node
        self.value = value
    
    def is_leaf(self):
        """
        Returns True if this node is a leaf.

        Implementation detail:
        - leaf nodes store a prediction in `value`
        - internal nodes have value=None and instead store feature/threshold and children
        """
        return self.value is not None


# -----------------------------
# Decision Tree Regressor (from scratch)
# -----------------------------
class DecisionTreeRegressorFromScratch:
    """
    A Decision Tree Regressor built from scratch (NumPy only).

    Goal:
    - Fit a tree that splits the feature space into regions (leaves),
      and predicts the mean target value in each region.

    This implementation uses:
    - greedy splitting: at each node, pick the split that most reduces MSE
    - axis-aligned splits: one feature at a time with a threshold
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        """
        Hyperparameters (how the tree is controlled):

        max_depth:
        - Maximum depth allowed for the tree.
        - Depth counts how many split "levels" you create from the root.
        - If max_depth is small, tree is simpler (less variance, more bias).
        - If max_depth is large/None, tree can become very complex (overfit easily).

        min_samples_split:
        - Minimum number of samples required at a node to even consider splitting it.
        - If a node has fewer than this number, it becomes a leaf.
        - Increasing it prevents tiny nodes from splitting, reducing overfitting.

        min_samples_leaf:
        - Minimum number of samples that must end up in EACH child after a split.
        - This is stricter than min_samples_split because it checks split validity.
        - Helps avoid splits that create very small leaves (high variance).

        min_impurity_decrease:
        - Minimum required improvement (reduction) in impurity to accept a split.
        - Here impurity is MSE, so "gain" = parent_mse - weighted_child_mse.
        - If the best gain is smaller than this, do not split (make a leaf).
        - Increasing it makes the tree more conservative (fewer splits).
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        # root will hold the top Node after fitting
        self.root = None

    def fit(self, X, y):
        """
        Train the tree on data (X, y).

        X: shape (n_samples, n_features)
        y: shape (n_samples,)

        This builds the entire tree recursively starting at the root.
        """
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _mse(self, y):
        """
        Compute Mean Squared Error (MSE) for a node.

        In a regression tree, the best constant prediction for a set of targets y
        is their mean. The MSE measures how "spread out" targets are around that mean.

        MSE = mean( (y_i - mean(y))^2 )

        Lower MSE = targets are more similar (a "purer" node).
        Splitting tries to reduce this quantity.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        """
        Find the best (feature, threshold) split at the current node.

        Greedy strategy:
        - Try every feature
        - For each feature, try candidate thresholds
        - Compute the MSE reduction ("gain")
        - Keep the split with the largest gain

        Returns:
        - (best_feature_index, best_threshold) if a valid split is found
        - (None, None) if no split meets constraints or improves enough
        """
        n_samples, n_features = X.shape

        # impurity of current node before splitting
        parent_mse = self._mse(y)

        # track best seen improvement and corresponding split parameters
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # loop over all features to consider splitting on each one
        for feature_idx in range(n_features):
            # take the column values for this feature across all samples
            X_col = X[:, feature_idx]

            # candidate thresholds are chosen as the unique values in the column
            # (note: this can be slow if many unique values; see note below)
            thresholds = np.unique(X_col)

            # try each threshold and evaluate split quality
            for t in thresholds:
                # split rule: left gets samples with feature <= t, right gets > t
                left_mask = X_col <= t
                right_mask = X_col > t

                # targets in each side
                y_left = y[left_mask]
                y_right = y[right_mask]

                # enforce min_samples_leaf: both children must have enough samples
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                n_left = len(y_left)
                n_right = len(y_right)

                # impurity in each child
                mse_left = self._mse(y_left)
                mse_right = self._mse(y_right)

                # weighted impurity after split (children impurity weighted by size)
                weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples

                # gain = how much MSE you reduced by splitting
                gain = parent_mse - weighted_mse

                # update best split if this is the largest gain so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = t

        # enforce min_impurity_decrease: reject weak splits
        # if even the best split doesn't improve enough, signal "no split"
        if best_gain < self.min_impurity_decrease:
            return None, None

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        At each call (node):
        1) Check stopping criteria -> if met, return a leaf Node(mean(y))
        2) Find best split (feature, threshold)
        3) If no valid split -> return leaf Node(mean(y))
        4) Otherwise split data and recursively build left/right subtrees
        """
        n_samples, n_features = X.shape  # n_features not used directly here, but included for completeness

        # ---- Stopping criteria ----
        # (A) depth limit reached
        # (B) not enough samples to split
        #
        # If we stop, we create a leaf predicting the mean of targets at this node.
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        # ---- Choose the best split ----
        feature, threshold = self._best_split(X, y)

        # If no split found (either no valid split or not enough impurity decrease),
        # make this a leaf node.
        if feature is None:
            return Node(value=np.mean(y))

        # ---- Split data using the chosen rule ----
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # ---- Recursively build children ----
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return an internal decision node containing split + pointers to children
        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    def _traverse_tree(self, x, node):
        """
        Predict for a single sample x by walking the tree from the given node.

        If node is a leaf -> return its stored value.
        Else:
          check x[node.feature] <= node.threshold
          go to left or right child accordingly
        """
        # base case: leaf node
        if node.is_leaf():
            return node.value

        # decision: choose branch based on the split rule
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """
        Predict for a batch of samples X.

        For each row x in X:
        - traverse the fitted tree from the root
        - collect the leaf prediction

        Returns:
        - numpy array of predictions, shape (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def save(self, path: str) -> str:
        """
        Save the decision tree model to a compressed .npz file.

        Args:
            path (str): Path to save the model

        Returns:
            str: Path where the model was saved
        """
        np.savez_compressed(
            path,
            # Hyperparameters
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,

            # Learned structure (recursive Node tree)
            root=np.array(self.root, dtype=object),
        )

        logger.info(f"DecisionTreeRegressorFromScratch saved to {path}")
        return path

"""
Extra notes (practical/implementation details):

1) Threshold choices:
   Using np.unique(feature_column) as thresholds is correct but can be very slow
   for continuous features with many unique values. Faster variants typically:
   - sort values once
   - only consider midpoints between consecutive unique sorted values
   - or use quantile-based candidate thresholds

2) Complexity:
   This implementation is closer to O(n_features * n_samples^2) in worst cases
   because for each node you try many thresholds and create masks repeatedly.

3) Works with PCA?
   Yes. PCA output is just numeric features (linear combinations). The tree will
   split on those components the same way it splits on any numeric features.
   The main tradeoff is interpretability: splits on principal components are
   harder to explain than splits on original features.
"""