import os
import numpy as np
import pickle
import time
import uuid
from my_random_forest import MyRandomForest
from gan_fraud_generator import generate_synthetic_data
from src.common.classification_summary import classification_summary
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager

def main():
    np.random.seed(42)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline1")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    gan_path = os.path.join(output_dir, "gan_weights.pkl")
    os.makedirs(output_dir, exist_ok=True)

    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting Pipeline 1 - Run ID: {run_id}")

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

        # Define model parameters
        baseline_params = {
            'n_trees': 150,
            'max_depth': 10,
            'min_samples_split': 10,
            'max_features': 'log2'
        }

        rf_baseline = MyRandomForest(**baseline_params)
        oob_error = rf_baseline.fit(X_train, y_train)
        training_time = time.time() - start_time

        val_preds = rf_baseline.predict(X_val)
        test_preds = rf_baseline.predict(X_test)
        val_probs = rf_baseline.predict_proba(X_val)
        test_probs = rf_baseline.predict_proba(X_test)

        metrics_val = classification_summary(y_val, val_preds, val_probs)
        metrics_test = classification_summary(y_test, test_preds, test_probs)

        print(
            f"Baseline RF - OOB Error: {oob_error:.4f}, Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

        baseline_path = os.path.join(output_dir, "baseline_rf.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(rf_baseline, f)

        db.store_pipeline_results(
            run_id, "pipeline1", "baseline_rf", baseline_params, 0.0,
            metrics_val, metrics_test, baseline_path, training_time
        )

        # Load GAN weights
        if not os.path.exists(gan_path):
            raise FileNotFoundError(f"GAN weights not found at {gan_path}. Run train_gan.py first.")
        with open(gan_path, "rb") as f:
            gan_data = pickle.load(f)
        g_weights = gan_data["g_weights"]
        mins = gan_data["mins"]
        maxs = gan_data["maxs"]

        # Augmented RF
        print("Training augmented Random Forest models...")
        synthetic_percents = [0.05, 0.10, 0.12, 0.15, 0.20, 0.30]
        fraud_idx = np.where(y_train == 1)[0]
        X_fraud = X_train[fraud_idx]

        for perc in synthetic_percents:
            print(f"  - Training with {int(perc * 100)}% synthetic data...")
            start_time = time.time()
            n_synth = int(len(X_fraud) * perc)
            X_synth = generate_synthetic_data(g_weights, n_synth, 64, n_features)
            X_synth_denorm = preprocessor.denormalize(X_synth, mins, maxs)
            y_synth = np.ones(len(X_synth_denorm), dtype=np.int64)

            X_aug = np.vstack([X_train, X_synth_denorm])
            y_aug = np.concatenate([y_train, y_synth])

            # Define augmented model parameters
            aug_params = {
                'n_trees': 50,
                'max_depth': 10,
                'min_samples_split': 15,
                'max_features': 'log2'
            }

            rf_aug = MyRandomForest(**aug_params)
            oob_error = rf_aug.fit(X_aug, y_aug)
            training_time = time.time() - start_time

            val_preds = rf_aug.predict(X_val)
            test_preds = rf_aug.predict(X_test)
            val_probs = rf_aug.predict_proba(X_val)
            test_probs = rf_aug.predict_proba(X_test)

            metrics_val = classification_summary(y_val, val_preds, val_probs)
            metrics_test = classification_summary(y_test, test_preds, test_probs)

            print(
                f"    Augmented RF OOB Error: {oob_error:.4f}, Val Accuracy: {metrics_val['accuracy']:.4f}, Test Accuracy: {metrics_test['accuracy']:.4f}")

            aug_path = os.path.join(output_dir, f"augmented_rf_{int(perc * 100)}.pkl")
            with open(aug_path, "wb") as f:
                pickle.dump(rf_aug, f)

            db.store_pipeline_results(
                run_id, "pipeline1", f"augmented_rf_{int(perc * 100)}", aug_params, perc,
                metrics_val, metrics_test, aug_path, training_time
            )

        db.log_message(run_id, "Pipeline 1 completed")
        db.close()

    except Exception as e:
        db.log_message(run_id, f"Pipeline 1 failed: {e}", "ERROR")
        db.close()
        raise

if __name__ == "__main__":
    main()