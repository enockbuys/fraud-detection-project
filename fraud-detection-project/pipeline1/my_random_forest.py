import numpy as np
from my_decision_tree import MyDecisionTree

class MyRandomForest:
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=10, max_features='log2'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaNs")
        n_samples, n_features = X.shape
        self.trees = []

        #how many features to sample for each tree
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = max(1, int(np.log2(n_features)) + 1)
        else:
            max_features = n_features

        #out-of-bag votes
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_trees):
            idx = np.random.choice(n_samples, n_samples, replace=True)#Loop to build each tree
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(idx))
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_subset = X[idx][:, feature_indices]
            tree = MyDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y[idx])

            self.trees.append((tree, feature_indices))

            if len(oob_idx) > 0:
                X_oob = X[oob_idx][:, feature_indices]
                oob_preds = tree.predict(X_oob).astype(np.float64)
                oob_predictions[oob_idx] += oob_preds
                oob_counts[oob_idx] += 1

        oob_mask = oob_counts > 0
        oob_error = 0
        if np.sum(oob_mask) > 0:
            oob_predictions[oob_mask] = (oob_predictions[oob_mask] / oob_counts[oob_mask]).round().astype(np.int64)
            oob_error = np.mean(oob_predictions[oob_mask] != y[oob_mask])

        return oob_error

    def predict(self, X):
        X = np.asarray(X)
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaNs")
        all_preds = np.array([tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees]).T
        return np.array([np.bincount(sample_preds).argmax() for sample_preds in all_preds])

    def predict_proba(self, X):
        X = np.asarray(X)
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaNs")
        all_preds = np.array([tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees]).T
        return np.mean(all_preds, axis=1)