import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import hashlib
import os
import json

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._create_tables()

    def _connect(self):
        if self.conn is None or self.cursor is None:
            try:
                self.conn = sqlite3.connect(self.db_path)
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                raise Exception(f"Database connection failed: {e}")

    def _create_tables(self):
        self._connect()

        # Raw data tracking
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE,
                file_path TEXT,
                record_count INTEGER,
                load_timestamp TEXT
            )
        ''')

        # Cleaned data summary
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cleaned_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_count INTEGER,
                feature_count INTEGER,
                fraud_count INTEGER,
                legit_count INTEGER,
                balance_strategy TEXT,
                preprocess_timestamp TEXT
            )
        ''')

        # Enhanced pipeline results with parameters and comprehensive metrics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                pipeline_name TEXT,
                model_type TEXT,
                model_params TEXT,  -- JSON string of model parameters
                synthetic_percent REAL,
                training_time REAL,
                -- Validation metrics
                val_accuracy REAL,
                val_precision REAL,
                val_recall REAL,
                val_f1_score REAL,
                val_roc_auc REAL,
                val_pr_auc REAL,
                val_true_positives INTEGER,
                val_false_positives INTEGER,
                val_false_negatives INTEGER,
                val_true_negatives INTEGER,
                -- Test metrics
                test_accuracy REAL,
                test_precision REAL,
                test_recall REAL,
                test_f1_score REAL,
                test_roc_auc REAL,
                test_pr_auc REAL,
                test_true_positives INTEGER,
                test_false_positives INTEGER,
                test_false_negatives INTEGER,
                test_true_negatives INTEGER,
                -- Additional info
                model_path TEXT,
                run_timestamp TEXT
            )
        ''')

        # Enhanced GAN training records
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS gan_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                pipeline_name TEXT,
                gan_type TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                latent_dim INTEGER,
                learning_rate REAL,
                final_g_loss REAL,
                final_d_loss REAL,
                synthetic_quality REAL,
                mean_diff REAL,
                std_diff REAL,
                kl_div REAL,
                model_path TEXT,
                training_time REAL,
                run_timestamp TEXT
            )
        ''')

        # Training logs
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                pipeline_name TEXT,
                log_level TEXT,
                message TEXT,
                timestamp TEXT
            )
        ''')

        self.conn.commit()

    def _get_file_hash(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            self.log_message("DATABASE", f"Error generating file hash: {e}", "ERROR")
            return None

    def store_raw_data(self, csv_file):
        self._connect()
        file_hash = self._get_file_hash(csv_file)
        if file_hash is None:
            return False

        timestamp = datetime.now().isoformat()
        self.cursor.execute('SELECT id FROM raw_data WHERE file_hash = ?', (file_hash,))
        if self.cursor.fetchone() is not None:
            self.log_message("DATABASE", "Raw data already exists, skipping", "INFO")
            return False

        try:
            df = pd.read_csv(csv_file)
            record_count = len(df)
            self.cursor.execute('''
                INSERT INTO raw_data (file_hash, file_path, record_count, load_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (file_hash, csv_file, record_count, timestamp))
            self.conn.commit()
            self.log_message("DATABASE", f"Raw data stored: {record_count} records", "INFO")
            return True
        except Exception as e:
            self.log_message("DATABASE", f"Error storing raw data: {e}", "ERROR")
            return False

    def store_cleaned_data_summary(self, X, y, balance_strategy):
        self._connect()
        timestamp = datetime.now().isoformat()
        try:
            fraud_count = np.sum(y == 1)
            legit_count = np.sum(y == 0)
            self.cursor.execute('''
                INSERT INTO cleaned_data (sample_count, feature_count, fraud_count, legit_count, balance_strategy, preprocess_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (len(X), X.shape[1], fraud_count, legit_count, balance_strategy, timestamp))
            self.conn.commit()
            return True
        except Exception as e:
            self.log_message("DATABASE", f"Error storing cleaned data summary: {e}", "ERROR")
            return False

    def log_message(self, run_id, message, level="INFO"):
        self._connect()
        timestamp = datetime.now().isoformat()
        try:
            self.cursor.execute('''
                INSERT INTO training_logs (run_id, pipeline_name, log_level, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (run_id, "SYSTEM", level, message, timestamp))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error logging message: {e}")
            return False

    def store_pipeline_results(self, run_id, pipeline_name, model_type, model_params, synthetic_percent,
                             metrics_val, metrics_test, model_path, training_time=0):
        self._connect()
        timestamp = datetime.now().isoformat()
        try:
            # Convert model_params to JSON string if it's a dict
            if isinstance(model_params, dict):
                model_params_json = json.dumps(model_params)
            else:
                model_params_json = str(model_params)

            self.cursor.execute('''
                INSERT INTO pipeline_results (
                    run_id, pipeline_name, model_type, model_params, synthetic_percent, training_time,
                    val_accuracy, val_precision, val_recall, val_f1_score, val_roc_auc, val_pr_auc,
                    val_true_positives, val_false_positives, val_false_negatives, val_true_negatives,
                    test_accuracy, test_precision, test_recall, test_f1_score, test_roc_auc, test_pr_auc,
                    test_true_positives, test_false_positives, test_false_negatives, test_true_negatives,
                    model_path, run_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, pipeline_name, model_type, model_params_json, synthetic_percent, training_time,
                metrics_val['accuracy'], metrics_val['precision'], metrics_val['recall'], metrics_val['f1_score'],
                metrics_val.get('roc_auc', 0), metrics_val.get('pr_auc', 0),
                metrics_val.get('true_positives', 0), metrics_val.get('false_positives', 0),
                metrics_val.get('false_negatives', 0), metrics_val.get('true_negatives', 0),
                metrics_test['accuracy'], metrics_test['precision'], metrics_test['recall'], metrics_test['f1_score'],
                metrics_test.get('roc_auc', 0), metrics_test.get('pr_auc', 0),
                metrics_test.get('true_positives', 0), metrics_test.get('false_positives', 0),
                metrics_test.get('false_negatives', 0), metrics_test.get('true_negatives', 0),
                model_path, timestamp
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.log_message(run_id, f"Error storing pipeline results: {e}", "ERROR")
            return False

    def store_gan_training(self, run_id, pipeline_name, gan_type, params, final_losses,
                          synthetic_quality, mean_diff, std_diff, kl_div, model_path, training_time=0):
        self._connect()
        timestamp = datetime.now().isoformat()
        try:
            self.cursor.execute('''
                INSERT INTO gan_training (
                    run_id, pipeline_name, gan_type, epochs, batch_size, latent_dim, learning_rate,
                    final_g_loss, final_d_loss, synthetic_quality, mean_diff, std_diff, kl_div,
                    model_path, training_time, run_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, pipeline_name, gan_type,
                params.get('epochs'), params.get('batch_size'), params.get('latent_dim'),
                params.get('learning_rate', params.get('g_lr', 0.0002)),
                final_losses.get('g_loss', 0), final_losses.get('d_loss', 0),
                synthetic_quality, mean_diff, std_diff, kl_div,
                model_path, training_time, timestamp
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.log_message(run_id, f"Error storing GAN training results: {e}", "ERROR")
            return False

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None