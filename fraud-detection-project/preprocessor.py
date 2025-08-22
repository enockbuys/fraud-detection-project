import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, csv_file):
        self.dataFrame = pd.read_csv(csv_file)
    def clean_and_split(self):
        cleanedDataFrame = self.dataFrame.dropna().drop_duplicates()

        if 'Time' in cleanedDataFrame.columns:
            cleanedDataFrame['Hour'] = (cleanedDataFrame['Time'] / 3600) % 24

        if 'Amount' in cleanedDataFrame.columns:
            cleanedDataFrame['log_amount'] = np.log1p(cleanedDataFrame['Amount'])

        X = cleanedDataFrame.drop(columns=['Class']).values
        y = cleanedDataFrame['Class'].values

        fraud_indexes = np.where(y == 1)[0]
        legitimate_indexes = np.where(y == 0)[0]

        print("Original fraud cases:", len(fraud_indexes))
        print("Original legit cases:", len(legitimate_indexes))

        np.random.shuffle(legitimate_indexes)
        number_of_frauds = len(fraud_indexes)
        balanced_legitimate_indexes = legitimate_indexes[:number_of_frauds]

        balanced_indexes = np.concatenate([fraud_indexes, balanced_legitimate_indexes])
        np.random.shuffle(balanced_indexes)

        X_balanced = X[balanced_indexes]
        y_balanced = y[balanced_indexes]

        total_balanced_samples = len(X_balanced)
        train_split_point = int(0.8 * total_balanced_samples)
        train_indexes = np.arange(train_split_point)
        test_indexes = np.arange(train_split_point, total_balanced_samples)

        validation_split_point = int(0.8 * len(train_indexes))
        validation_indexes = train_indexes[validation_split_point:]
        final_training_indexes = train_indexes[:validation_split_point]

        X_train = X_balanced[final_training_indexes]
        y_train = y_balanced[final_training_indexes]

        X_validation = X_balanced[validation_indexes]
        y_validation = y_balanced[validation_indexes]

        X_test = X_balanced[test_indexes]
        y_test = y_balanced[test_indexes]

        print("Fraud cases in training:", np.sum(y_train))
        print("Fraud cases in validation:", np.sum(y_validation))
        print("Fraud cases in test:", np.sum(y_test))
        print("Total training samples:", len(y_train))

        return X_train, y_train, X_validation, y_validation, X_test, y_test