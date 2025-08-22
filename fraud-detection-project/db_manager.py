import sqlite3
import numpy as np

class DBManager:
    def __init__(self, database_name="fraud_detection.db"):
        self.connection = sqlite3.connect(database_name)
        self.cursor = self.connection.cursor()

    def save_split(self, dataset_id, split_type, features_array, labels_array):
        self.cursor.execute("INSERT INTO splits (dataset_id, split_type, features, labels) VALUES (?, ?, ?, ?)",(dataset_id,split_type,features_array.tobytes(),labels_array.tobytes()))
        self.connection.commit()

    def load_split(self, dataset_id, split_type, number_of_features):
        self.cursor.execute("SELECT features, labels FROM splits WHERE dataset_id=? AND split_type=?",(dataset_id, split_type))
        row = self.cursor.fetchone()
        if row:
            features_array = np.frombuffer(row[0], dtype=np.float64).reshape(-1, number_of_features)
            labels_array = np.frombuffer(row[1], dtype=np.int64)
            return features_array, labels_array
        else:
            return None, None

    def save_predictions(self, model_name, dataset_id, split_type, true_labels_array, predicted_labels_array):
        self.cursor.execute("INSERT INTO predictions (model_name, dataset_id, split_type, true_labels, predicted_labels) VALUES (?, ?, ?, ?, ?)",
            (model_name,dataset_id,split_type,true_labels_array.tobytes(),predicted_labels_array.tobytes()))
        self.connection.commit()

    def log_event(self, model_name, action_taken, event_message):
        self.cursor.execute("INSERT INTO logs (model_name, action, message) VALUES (?, ?, ?)",(model_name,action_taken,event_message))
        self.connection.commit()

    def close(self):
        self.connection.close()