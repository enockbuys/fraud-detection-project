import sys
import os
import numpy as np
import pickle
import time
from gan_fraud_generator import train_gan, generate_synthetic_data, evaluate_samples
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager
import uuid

def main():
    np.random.seed(42)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline1")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    os.makedirs(output_dir, exist_ok=True)

    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting GAN training - Run ID: {run_id}")

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

        X_fraud_train_norm, mins, maxs = preprocessor.normalize(X_fraud)
        X_fraud_val_norm = preprocessor.normalize(X_fraud_val)[0]

        print(f"Fraud samples for GAN training: {len(X_fraud)}")
        print(f"Shape of X_fraud_train_norm: {X_fraud_train_norm.shape}")
        start_time = time.time()

        g_weights, training_history, final_losses = train_gan(
            real_data=X_fraud_train_norm,
            val_data=X_fraud_val_norm,
            epochs=200,
            batch_size=256,
            latent_dim=64,
            feature_dim=n_features,
            g_lr=2e-4,
            d_lr=2e-4,
            n_critic=5,
            lambda_gp=10.0,
            print_interval=25
        )

        gan_training_time = time.time() - start_time
        gan_path = os.path.join(output_dir, "gan_weights.pkl")
        with open(gan_path, "wb") as f:
            pickle.dump({"g_weights": g_weights, "mins": mins, "maxs": maxs}, f)

        synthetic_test = generate_synthetic_data(g_weights, 1000, 64, n_features)
        mean_diff, std_diff, kl_div = evaluate_samples(X_fraud_val_norm, synthetic_test)
        synthetic_quality = 1.0 / (1.0 + kl_div)

        print(f"\nGAN Training Completed!")
        print(f"Training Time: {gan_training_time:.2f} seconds")
        print(f"GAN Quality - Mean Diff: {mean_diff:.4f}, Std Diff: {std_diff:.4f}, KL Div: {kl_div:.4f}")

        # Store GAN training results with enhanced parameters
        gan_params = {
            'epochs': 200,
            'batch_size': 256,
            'latent_dim': 64,
            'learning_rate': 2e-4,
            'n_critic': 5,
            'lambda_gp': 10.0
        }

        db.store_gan_training(
            run_id, "pipeline1", "custom_gan", gan_params,
            final_losses, synthetic_quality, mean_diff, std_diff, kl_div,
            gan_path, gan_training_time
        )

        db.log_message(run_id, "GAN training completed")
        db.close()

    except Exception as e:
        print(f"Error occurred: {e}")
        db.log_message(run_id, f"GAN training failed: {e}", "ERROR")
        db.close()
        raise

if __name__ == "__main__":
    main()