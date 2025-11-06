import os
import numpy as np
import pickle
import time
from ctgan import CTGAN
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager
from scipy.stats import wasserstein_distance
import uuid

def evaluate_samples(real_data, fake_data):
    #Evaluate synthetic data quality using Wasserstein distance.
    mean_diff = np.mean(np.abs(np.mean(real_data, axis=0) - np.mean(fake_data, axis=0)))
    std_diff = np.mean(np.abs(np.std(real_data, axis=0) - np.std(fake_data, axis=0)))
    kl_div = np.mean([wasserstein_distance(real_data[:, i], fake_data[:, i]) for i in range(real_data.shape[1])])
    return mean_diff, std_diff, kl_div

def main():
    np.random.seed(42)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline3")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    os.makedirs(output_dir, exist_ok=True)

    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting CTGAN training - Run ID: {run_id}")

    try:
        print("Loading and preprocessing data...")
        preprocessor = Preprocessor(data_path)
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocessor.clean_and_split(balance_data=False)
        n_features = X_train.shape[1]
        print(f"Number of features: {n_features}")

        db.store_raw_data(data_path)
        db.store_cleaned_data_summary(X_train, y_train, balance_strategy="noSMOTE")

        fraud_idx = np.where(y_train == 1)[0]
        X_fraud = X_train[fraud_idx[:10000]]
        fraud_val_idx = np.where(y_val == 1)[0]
        X_fraud_val = X_val[fraud_val_idx]

        print(f"Fraud samples for CTGAN training: {len(X_fraud)}")
        start_time = time.time()

        ctgan = CTGAN(
            epochs=200,
            batch_size=250,
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            verbose=True
        )
        ctgan.fit(X_fraud)

        gan_training_time = time.time() - start_time
        gan_path = os.path.join(output_dir, "ctgan_model.pkl")
        with open(gan_path, "wb") as f:
            pickle.dump(ctgan, f)

        synthetic_test = ctgan.sample(1000)
        mean_diff, std_diff, kl_div = evaluate_samples(X_fraud_val, synthetic_test)
        synthetic_quality = 1.0 / (1.0 + kl_div)

        print(f"\nCTGAN Training Completed!")
        print(f"Training Time: {gan_training_time:.2f} seconds")
        print(f"CTGAN Quality - Mean Diff: {mean_diff:.4f}, Std Diff: {std_diff:.4f}, KL Div: {kl_div:.4f}")

        # Store CTGAN training results
        gan_params = {
            'epochs': 200,
            'batch_size': 250,
            'generator_lr': 2e-4,
            'discriminator_lr': 2e-4
        }

        db.store_gan_training(
            run_id, "pipeline3", "ctgan", gan_params,
            {"g_loss": None, "d_loss": None}, synthetic_quality, mean_diff, std_diff, kl_div,
            gan_path, gan_training_time
        )

        db.log_message(run_id, "CTGAN training completed")
        db.close()

    except Exception as e:
        print(f"Error occurred: {e}")
        db.log_message(run_id, f"CTGAN training failed: {e}", "ERROR")
        db.close()
        raise

if __name__ == "__main__":
    main()