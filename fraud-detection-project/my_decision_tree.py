import numpy as np
class MyDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_structure = None

    def fit(self, X, y):
        y = y.astype(np.int64)
        self.tree_structure = self._build_tree(X, y, current_depth=0)

    def _build_tree(self, X, y, current_depth):
        number_of_samples, number_of_features = X.shape
        number_of_classes = len(np.unique(y))

        if (current_depth >= self.max_depth) or (number_of_classes == 1) or (number_of_samples < self.min_samples_split):
            return self._majority_class(y)

        best_feature_index = None
        best_split_threshold = None
        best_gini_value = 1.0
        best_split_data = None

        for feature_index in range(number_of_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                left_side_mask = feature_values <= threshold
                right_side_mask = feature_values > threshold

                if np.sum(left_side_mask) == 0 or np.sum(right_side_mask) == 0:
                    continue

                left_side_labels = y[left_side_mask]
                right_side_labels = y[right_side_mask]

                gini_value = self._calculate_gini(left_side_labels, right_side_labels)

                if gini_value < best_gini_value:
                    best_gini_value = gini_value
                    best_feature_index = feature_index
                    best_split_threshold = threshold
                    best_split_data = {'left_X': X[left_side_mask],'left_y': y[left_side_mask],'right_X': X[right_side_mask],'right_y': y[right_side_mask]}

        if best_feature_index is None:
            return self._majority_class(y)

        left_subtree = self._build_tree(best_split_data['left_X'], best_split_data['left_y'], current_depth + 1)
        right_subtree = self._build_tree(best_split_data['right_X'], best_split_data['right_y'], current_depth + 1)

        decision_node = {'feature_index': best_feature_index,'threshold': best_split_threshold,'left_branch': left_subtree,'right_branch': right_subtree}
        return decision_node

    def _calculate_gini(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)

        def gini(labels):
            if len(labels) == 0:
                return 0
            unique_classes = np.unique(labels)
            score = 0
            for single_class in unique_classes:
                proportion = np.sum(labels == single_class) / len(labels)
                score += proportion ** 2
            return 1 - score

        gini_left = gini(left_labels)
        gini_right = gini(right_labels)

        weighted_gini = (len(left_labels) / total_samples) * gini_left + \
                        (len(right_labels) / total_samples) * gini_right
        return weighted_gini

    def _majority_class(self, y):
        y_int = y.astype(np.int64)
        class_counts = np.bincount(y_int)
        return np.argmax(class_counts)

    def predict(self, X):
        predictions_list = []
        for single_row in X:
            prediction = self._predict_one(single_row, self.tree_structure)
            predictions_list.append(prediction)
        return np.array(predictions_list)

    def _predict_one(self, single_row, node):
        if not isinstance(node, dict):
            return node

        feature_value = single_row[node['feature_index']]

        if feature_value <= node['threshold']:
            return self._predict_one(single_row, node['left_branch'])
        else:
            return self._predict_one(single_row, node['right_branch'])