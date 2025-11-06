import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import uuid
from src.common.classification_summary import classification_summary
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager
from train_gan_tf import generate_synthetic_data_tf


def main():
    np.random.seed(42)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU detected. Running on CPU.")
    else:
        print(f"Using GPU: {gpus}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline2")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    gan_path = os.path.join(output_dir, "gan_generator.keras")
    gan_params_path = os.path.join(output_dir, "gan_params.pkl")
    os.makedirs(output_dir, exist_ok=True)

    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting Pipeline 2 - Run ID: {run_id}")

    try:
        preprocessor = Preprocessor(data_path)
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocessor.clean_and_split(balance_data=False)
        n_features = X_train.shape[1]

        db.store_raw_data(data_path)
        db.store_cleaned_data_summary(X_train, y_train, balance_strategy="noSMOTE")

        print(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, Features={n_features}")

        # Baseline RF
        print("Training baseline Random Forest...")
        start_time = time.time()

        # Define baseline parameters
        baseline_params = {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 10,
            'max_features': 'log2',
            'random_state': 42,
            'n_jobs': -1
        }

        rf_baseline = RandomForestClassifier(**baseline_params)
        rf_baseline.fit(X_train, y_train)
        training_time = time.time() - start_time

        val_preds = rf_baseline.predict(X_val)
        test_preds = rf_baseline.predict(X_test)
        val_probs = rf_baseline.predict_proba(X_val)[:, 1]
        test_probs = rf_baseline.predict_proba(X_test)[:, 1]

        metrics_val = classification_summary(y_val, val_preds, val_probs)
        metrics_test = classification_summary(y_test, test_preds, test_probs)

        print(
            f"Baseline RF - Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

        baseline_path = os.path.join(output_dir, "baseline_rf_sklearn.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(rf_baseline, f)

        db.store_pipeline_results(
            run_id, "pipeline2", "baseline_rf_sklearn", baseline_params, 0.0,
            metrics_val, metrics_test, baseline_path, training_time
        )

        # Load GAN
        if not os.path.exists(gan_path) or not os.path.exists(gan_params_path):
            raise FileNotFoundError(
                f"GAN weights or params not found at {gan_path}/{gan_params_path}. Run train_gan_tf.py first.")
        generator = tf.keras.models.load_model(gan_path)
        with open(gan_params_path, "rb") as f:
            gan_params = pickle.load(f)
        mins = gan_params["mins"]
        maxs = gan_params["maxs"]

        # Augmented RF
        print("Training augmented Random Forest models...")
        synthetic_percents = [0.05, 0.10, 0.12, 0.15, 0.20, 0.30]
        fraud_idx = np.where(y_train == 1)[0]
        X_fraud = X_train[fraud_idx]

        for perc in synthetic_percents:
            print(f"  - Training with {int(perc * 100)}% synthetic data...")
            start_time = time.time()
            n_synth = int(len(X_fraud) * perc)
            X_synth = generate_synthetic_data_tf(generator, n_synth, 64, n_features)
            X_synth_denorm = preprocessor.denormalize(X_synth, mins, maxs)
            y_synth = np.ones(len(X_synth_denorm), dtype=np.int64)

            X_aug = np.vstack([X_train, X_synth_denorm])
            y_aug = np.concatenate([y_train, y_synth])

            # Define augmented parameters
            aug_params = {
                'n_estimators': 150,
                'max_depth': 10,
                'min_samples_split': 10,
                'max_features': 'log2',
                'random_state': 42,
                'n_jobs': -1
            }

            rf_aug = RandomForestClassifier(**aug_params)
            rf_aug.fit(X_aug, y_aug)
            training_time = time.time() - start_time

            val_preds = rf_aug.predict(X_val)
            test_preds = rf_aug.predict(X_test)
            val_probs = rf_aug.predict_proba(X_val)[:, 1]
            test_probs = rf_aug.predict_proba(X_test)[:, 1]

            metrics_val = classification_summary(y_val, val_preds, val_probs)
            metrics_test = classification_summary(y_test, test_preds, test_probs)

            print(
                f"    Augmented RF - Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

            aug_path = os.path.join(output_dir, f"augmented_rf_{int(perc * 100)}.pkl")
            with open(aug_path, "wb") as f:
                pickle.dump(rf_aug, f)

            db.store_pipeline_results(
                run_id, "pipeline2", f"augmented_rf_{int(perc * 100)}", aug_params, perc,
                metrics_val, metrics_test, aug_path, training_time
            )

        db.log_message(run_id, "Pipeline 2 completed")
        db.close()

    except Exception as e:
        db.log_message(run_id, f"Pipeline 2 failed: {e}", "ERROR")
        db.close()
        raise


if __name__ == "__main__":
    main()