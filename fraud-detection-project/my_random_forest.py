import numpy as np
from my_decision_tree import MyDecisionTree
class MyRandomForest:
    def __init__(self, number_of_trees=5, max_depth=5, min_samples_split=2):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.list_of_trees = []

    def fit(self, X, y):
        self.list_of_trees = []

        for tree_number in range(self.number_of_trees):
            bootstrap_indexes = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap_sample = X[bootstrap_indexes]
            y_bootstrap_sample = y[bootstrap_indexes]

            decision_tree = MyDecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            decision_tree.fit(X_bootstrap_sample, y_bootstrap_sample)

            self.list_of_trees.append(decision_tree)

    def predict(self, X):

        predictions_from_all_trees = []
        for single_tree in self.list_of_trees:
            predictions = single_tree.predict(X)
            predictions_from_all_trees.append(predictions)

        predictions_from_all_trees = np.array(predictions_from_all_trees).T

        final_predictions_list = []
        for predictions_for_one_sample in predictions_from_all_trees:
            class_counts = np.bincount(predictions_for_one_sample)
            majority_class = np.argmax(class_counts)
            final_predictions_list.append(majority_class)

        return np.array(final_predictions_list)