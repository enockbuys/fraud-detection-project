import numpy as np
import tensorflow as tf
import os
import time
from keras.saving import register_keras_serializable
from scipy.stats import wasserstein_distance
from src.common.preprocessor import Preprocessor
from src.common.database_manager import DatabaseManager
import uuid
import pickle

def normalize(data, mins, maxs):
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (data - mins) / ranges * 2 - 1

@register_keras_serializable()
class Generator(tf.keras.Model):
    def __init__(self, latent_dim, feature_dim, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(feature_dim, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'feature_dim': self.feature_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            latent_dim=config['latent_dim'],
            feature_dim=config['feature_dim']
        )


@register_keras_serializable()
class Critic(tf.keras.Model):
    def __init__(self, feature_dim, *args, **kwargs):
        super(Critic, self).__init__(*args, **kwargs)
        self.feature_dim = feature_dim
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def get_config(self):
        return {
            'feature_dim': self.feature_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            feature_dim=config['feature_dim']
        )

@tf.function
def compute_gradient_penalty(critic, real_data, fake_data, lambda_gp):
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        critic_inter = critic(interpolates)
    gradients = tape.gradient(critic_inter, [interpolates])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
    penalty = lambda_gp * tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty

def evaluate_samples(real_data, fake_data):#Compares generated data to real data
    n_eval = min(1000, len(real_data), len(fake_data))
    real_sample = real_data[:n_eval]
    fake_sample = fake_data[:n_eval]
    mean_diff = np.mean(np.abs(np.mean(real_sample, axis=0) - np.mean(fake_sample, axis=0)))
    std_diff = np.mean(np.abs(np.std(real_sample, axis=0) - np.std(fake_sample, axis=0)))
    kl_div = np.mean([wasserstein_distance(real_sample[:, i], fake_sample[:, i])
                      for i in range(min(10, real_sample.shape[1]))])
    return mean_diff, std_diff, kl_div

def generate_synthetic_data_tf(generator, n_samples, latent_dim, feature_dim):
    noise = tf.random.normal([n_samples, latent_dim])
    return generator(noise).numpy()

@tf.function
def train_step(generator, critic, g_optimizer, c_optimizer, real_batch, latent_dim, n_critic, lambda_gp):
    batch_size = tf.shape(real_batch)[0]
    for _ in range(n_critic):
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as c_tape:
            fake_data = generator(noise, training=True)
            c_real = critic(real_batch, training=True)
            c_fake = critic(fake_data, training=True)
            wass_loss = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real)
            penalty = compute_gradient_penalty(critic, real_batch, fake_data, lambda_gp)
            c_loss = wass_loss + penalty
        c_grads = c_tape.gradient(c_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(c_grads, critic.trainable_variables))
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as g_tape:
        fake_data = generator(noise, training=True)
        c_fake = critic(fake_data, training=True)
        g_loss = -tf.reduce_mean(c_fake)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    return c_loss, g_loss, wass_loss, penalty

def train_gan_tf(real_data, val_data=None, epochs=500, batch_size=256, latent_dim=128, feature_dim=30,
                 g_lr=0.0001, d_lr=0.0001, n_critic=5, lambda_gp=10.0, print_interval=50):
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    generator = Generator(latent_dim, feature_dim)
    critic = Critic(feature_dim)
    g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.5, beta_2=0.9)
    c_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.5, beta_2=0.9)
    real_data_tensor = tf.convert_to_tensor(real_data, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(real_data_tensor)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    training_history = {'c_loss': [], 'g_loss': [], 'mean_diff': [], 'std_diff': [], 'kl_div': []}
    best_kl = np.inf
    patience = 100
    wait = 0
    print(f"Starting training with {len(real_data)} samples, {len(list(dataset))} batches per epoch")
    for epoch in range(epochs):
        epoch_c_loss = 0
        epoch_g_loss = 0
        epoch_wass_loss = 0
        epoch_penalty = 0
        num_batches = 0
        for batch in dataset:
            if tf.shape(batch)[0] < 2:
                continue
            c_loss, g_loss, wass_loss, penalty = train_step(
                generator, critic, g_optimizer, c_optimizer, batch,
                latent_dim, n_critic, lambda_gp
            )
            epoch_c_loss += c_loss.numpy()
            epoch_g_loss += g_loss.numpy()
            epoch_wass_loss += wass_loss.numpy()
            epoch_penalty += penalty.numpy()
            num_batches += 1
        if num_batches == 0:
            continue
        avg_c_loss = epoch_c_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_wass_loss = epoch_wass_loss / num_batches
        avg_penalty = epoch_penalty / num_batches
        if epoch % print_interval == 0:
            test_samples = generate_synthetic_data_tf(generator, min(1000, len(real_data)), latent_dim, feature_dim)
            mean_diff, std_diff, kl_div = evaluate_samples(real_data, test_samples)
            training_history['c_loss'].append(avg_c_loss)
            training_history['g_loss'].append(avg_g_loss)
            training_history['mean_diff'].append(mean_diff)
            training_history['std_diff'].append(std_diff)
            training_history['kl_div'].append(kl_div)
            val_kl = kl_div
            if val_data is not None:
                val_mean_diff, val_std_diff, val_kl = evaluate_samples(val_data, test_samples)
            print(f"Epoch {epoch}: C Loss: {avg_c_loss:.4f}, G Loss: {avg_g_loss:.4f}, "
                  f"Wass: {avg_wass_loss:.4f}, Penalty: {avg_penalty:.4f}, Val KL: {val_kl:.4f}")
            if val_data is not None:
                if val_kl < best_kl:
                    best_kl = val_kl
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
    final_losses = {'g_loss': avg_g_loss, 'd_loss': avg_c_loss}
    return generator, training_history, final_losses

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    db_path = os.path.join(parent_dir, "fraud_detection.db")
    output_dir = os.path.join(parent_dir, "results", "pipeline2")
    data_path = os.path.join(parent_dir, "data", "raw", "creditcard.csv")
    os.makedirs(output_dir, exist_ok=True)
    db = DatabaseManager(db_path)
    run_id = str(uuid.uuid4())
    db.log_message(run_id, f"Starting GAN training (TF) - Run ID: {run_id}")
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
        print(f"Fraud samples for GAN training: {len(X_fraud)}")
        X_fraud_train_norm, mins, maxs = preprocessor.normalize(X_fraud)
        X_fraud_val_norm = normalize(X_fraud_val, mins, maxs)
        print("Starting GAN training with optimized settings...")
        start_time = time.time()
        generator, training_history, final_losses = train_gan_tf(
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
        gan_path = os.path.join(output_dir, "gan_generator.keras")
        generator.save(gan_path)
        synthetic_test = generate_synthetic_data_tf(generator, 1000, 64, n_features)
        mean_diff, std_diff, kl_div = evaluate_samples(X_fraud_val_norm, synthetic_test)
        synthetic_quality = 1.0 / (1.0 + kl_div)
        print(f"\nGAN Training Completed!")
        print(f"Training Time: {gan_training_time:.2f} seconds")
        print(f"GAN Quality - Mean Diff: {mean_diff:.4f}, Std Diff: {std_diff:.4f}, KL Div: {kl_div:.4f}")

        # Store enhanced GAN training results
        gan_params = {
            'epochs': 200,
            'batch_size': 256,
            'latent_dim': 64,
            'learning_rate': 2e-4,
            'n_critic': 5,
            'lambda_gp': 10.0
        }

        db.store_gan_training(
            run_id, "pipeline2", "tensorflow_wgan", gan_params,
            final_losses, synthetic_quality, mean_diff, std_diff, kl_div,
            gan_path, gan_training_time
        )

        with open(os.path.join(output_dir, "gan_params.pkl"), "wb") as f:
            pickle.dump({"mins": mins, "maxs": maxs}, f)
        db.log_message(run_id, "GAN training (TF) completed")
        db.close()
        print("GAN training completed successfully!")
    except Exception as e:
        db.log_message(run_id, f"GAN training (TF) failed: {e}", "ERROR")
        db.close()
        print(f"GAN training failed: {e}")
        raise

if __name__ == "__main__":
    main()