import numpy as np

class MyDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int64)
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaNs")
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            return self._majority_class(y)

        best_feature, best_threshold, best_gini, best_splits = None, None, 1.0, None

        feature_indices = np.random.permutation(n_features)
        for feature_index in feature_indices:
            unique_vals = np.unique(X[:, feature_index])
            if len(unique_vals) < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            if len(thresholds) > 50:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 50))

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                gini = self._gini(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
                    best_splits = {
                        'left_X': X[left_mask], 'left_y': left_y,
                        'right_X': X[right_mask], 'right_y': right_y
                    }

        if best_feature is None:
            return self._majority_class(y)

        left_subtree = self._build_tree(best_splits['left_X'], best_splits['left_y'], depth + 1)
        right_subtree = self._build_tree(best_splits['right_X'], best_splits['right_y'], depth + 1)

        return {'feature_index': best_feature, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}

    def _gini(self, left_y, right_y):
        def gini_impurity(labels):
            if len(labels) == 0:
                return 0
            unique, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return 1 - np.sum(probs ** 2)

        total = len(left_y) + len(right_y)
        if total == 0:
            return 0
        return (len(left_y) / total) * gini_impurity(left_y) + (len(right_y) / total) * gini_impurity(right_y)

    def _majority_class(self, y):
        if len(y) == 0:
            return np.array(0)
        return np.bincount(y.astype(np.int64)).argmax()

    def predict(self, X):
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaNs")
        X = np.asarray(X)
        preds = np.array([self._predict_single(x, self.tree) for x in X])
        return preds

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        branch = 'left' if x[node['feature_index']] <= node['threshold'] else 'right'
        return self._predict_single(x, node[branch])