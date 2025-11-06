import os
import numpy as np
import xgboost as xgb
import pickle
import time
import uuid
from src.common.classification_summary import classification_summary
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager

def main():
    np.random.seed(42)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline3")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    gan_path = os.path.join(output_dir, "ctgan_model.pkl")
    os.makedirs(output_dir, exist_ok=True)

    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting Pipeline 3 - Run ID: {run_id}")

    try:
        preprocessor = Preprocessor(data_path)
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocessor.clean_and_split(balance_data=False)
        n_features = X_train.shape[1]

        db.store_raw_data(data_path)
        db.store_cleaned_data_summary(X_train, y_train, balance_strategy="NoSMOTE")

        print(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, Features={n_features}")

        # Baseline XGBoost
        print("Training baseline XGBoost...")
        start_time = time.time()

        # Define baseline parameters
        baseline_params = {
            'n_estimators': 150,
            'max_depth': 10,
            'min_child_weight': 10,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        xgb_baseline = xgb.XGBClassifier(**baseline_params)
        xgb_baseline.fit(X_train, y_train)
        training_time = time.time() - start_time

        val_preds = xgb_baseline.predict(X_val)
        test_preds = xgb_baseline.predict(X_test)
        val_probs = xgb_baseline.predict_proba(X_val)[:, 1]
        test_probs = xgb_baseline.predict_proba(X_test)[:, 1]

        metrics_val = classification_summary(y_val, val_preds, val_probs)
        metrics_test = classification_summary(y_test, test_preds, test_probs)

        print(f"Baseline XGBoost - Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

        baseline_path = os.path.join(output_dir, "baseline_xgb.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(xgb_baseline, f)

        db.store_pipeline_results(
            run_id, "pipeline3", "baseline_xgb", baseline_params, 0.0,
            metrics_val, metrics_test, baseline_path, training_time
        )

        # Load CTGAN
        if not os.path.exists(gan_path):
            raise FileNotFoundError(f"CTGAN model not found at {gan_path}. Run train_gan_ctgan.py first.")
        with open(gan_path, "rb") as f:
            ctgan = pickle.load(f)

        # Augmented XGBoost
        print("Training augmented XGBoost models...")
        synthetic_percents = [0.05, 0.10, 0.12, 0.15, 0.20, 0.30]
        fraud_idx = np.where(y_train == 1)[0]
        X_fraud = X_train[fraud_idx]
        n_fraud = len(X_fraud)

        for perc in synthetic_percents:
            print(f"  - Training with {int(perc * 100)}% synthetic data...")
            start_time = time.time()
            n_synth = int(n_fraud * perc)
            X_synth = ctgan.sample(n_synth)
            y_synth = np.ones(n_synth, dtype=np.int64)

            X_aug = np.vstack([X_train, X_synth])
            y_aug = np.concatenate([y_train, y_synth])

            # Define augmented parameters
            aug_params = {
                'n_estimators': 150,
                'max_depth': 10,
                'min_child_weight': 10,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }

            xgb_aug = xgb.XGBClassifier(**aug_params)
            xgb_aug.fit(X_aug, y_aug)
            training_time = time.time() - start_time

            val_preds = xgb_aug.predict(X_val)
            test_preds = xgb_aug.predict(X_test)
            val_probs = xgb_aug.predict_proba(X_val)[:, 1]
            test_probs = xgb_aug.predict_proba(X_test)[:, 1]

            metrics_val = classification_summary(y_val, val_preds, val_probs)
            metrics_test = classification_summary(y_test, test_preds, test_probs)

            print(f"Augmented XGBoost - Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

            aug_path = os.path.join(output_dir, f"augmented_xgb_{int(perc * 100)}.pkl")
            with open(aug_path, "wb") as f:
                pickle.dump(xgb_aug, f)

            db.store_pipeline_results(
                run_id, "pipeline3", f"augmented_xgb_{int(perc * 100)}", aug_params, perc,
                metrics_val, metrics_test, aug_path, training_time
            )

        db.log_message(run_id, "Pipeline 3 completed")
        db.close()

    except Exception as e:
        db.log_message(run_id, f"Pipeline 3 failed: {e}", "ERROR")
        db.close()
        raise

if __name__ == "__main__":
    main()