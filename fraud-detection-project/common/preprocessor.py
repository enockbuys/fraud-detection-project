import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import os
class Preprocessor:
    def __init__(self, csv_file):
        try:
            self.dataFrame = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

    def clean_and_split(self, balance_data=True, train_ratio=0.7, validation_ratio=0.2, predict_ratio=0.05):
        if not (0 < train_ratio < 1) or not (0 < validation_ratio < 1) or not (0 < predict_ratio < 1):
            raise ValueError("Ratios must be between 0 and 1")
        if train_ratio + validation_ratio + predict_ratio > 1:
            raise ValueError("Sum of ratios must not exceed 1")

        df = self.dataFrame.dropna().drop_duplicates()
        if df.empty:
            raise ValueError("Dataset is empty after cleaning")

        required_columns = ['Class', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Dataset must contain 'Class', 'Time', 'Amount', and 'V1'–'V28' columns")

        # Feature engineering
        df['Hour'] = (df['Time'] % (24 * 3600)) / 3600
        df['log_amount'] = np.log1p(df['Amount'].clip(lower=0))
        X = df.drop(columns=['Class', 'Time', 'Amount']).values.astype(np.float32)
        y = df['Class'].values.astype(np.int64)

        # Split into predict and remaining sets
        n_samples = len(X)
        predict_size = int(predict_ratio * n_samples)
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        X_predict, y_predict = X[:predict_size], y[:predict_size]
        X_remain, y_remain = X[predict_size:], y[predict_size:]

        # Split remaining into train/val/test
        remain_samples = len(X_remain)
        train_size = int(train_ratio * remain_samples)
        val_size = int(validation_ratio * remain_samples)
        test_size = remain_samples - train_size - val_size

        indices = np.random.permutation(remain_samples)
        X_remain, y_remain = X_remain[indices], y_remain[indices]

        X_train = X_remain[:train_size]
        y_train = y_remain[:train_size]
        X_val = X_remain[train_size:train_size + val_size]
        y_val = y_remain[train_size:train_size + val_size]
        X_test = X_remain[train_size + val_size:]
        y_test = y_remain[train_size + val_size:]

        # Apply SMOTE for balancing
        if balance_data:
            try:
                smote = SMOTE(random_state=42, sampling_strategy=1.0)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                raise Exception(f"SMOTE failed: {e}")

        # Save predict set
        np.savez(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "predict_set.npz"),
                 X_predict=X_predict, y_predict=y_predict)

        return X_train, y_train, X_val, y_val, X_test, y_test, X_predict, y_predict

    def normalize(self, X):
        #Normalize data to [-1, 1] for GAN training.
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (X - mins) / ranges * 2 - 1, mins, maxs

    def denormalize(self, X, mins, maxs):
        #Denormalize data back to original scale.
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return ((X + 1) / 2) * ranges + mins
