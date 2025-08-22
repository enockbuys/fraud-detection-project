import numpy as np
class ConditionalGAN:
    def __init__(self, latent_dim=100, output_dim=33):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.G_W1 = np.random.normal(0, 0.02, (latent_dim + 1, 256))
        self.G_W2 = np.random.normal(0, 0.02, (256, 512))
        self.G_W3 = np.random.normal(0, 0.02, (512, output_dim))

        self.D_W1 = np.random.normal(0, 0.01, (output_dim + 1, 256))
        self.D_W2 = np.random.normal(0, 0.01, (256, 128))
        self.D_W3 = np.random.normal(0, 0.01, (128, 1))

        self.g_lr = 0.0005
        self.gradient_penalty_weight = 10

    def generator_forward(self, z, c):
        z = np.concatenate([z, c], axis=1)
        h1 = self.leaky_relu(np.dot(z, self.G_W1))
        h2 = self.leaky_relu(np.dot(h1, self.G_W2))
        return np.dot(h2, self.G_W3)

    def discriminator_forward(self, x, c):
        x = np.concatenate([x, c], axis=1)
        h1 = self.leaky_relu(np.dot(x, self.D_W1))
        h2 = self.leaky_relu(np.dot(h1, self.D_W2))
        return np.dot(h2, self.D_W3)

    def leaky_relu(self, x, alpha=0.2):
        return np.maximum(alpha * x, x)

    def train(self, X_real, y_real, epochs=100, batch_size=32):
        real_labels = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(5):
                idx = np.random.randint(0, X_real.shape[0], batch_size)
                real_batch = X_real[idx]
                condition = y_real[idx].reshape(-1, 1)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_batch = self.generator_forward(noise, condition)

                alpha = np.random.random((batch_size, 1))
                interpolated = alpha * real_batch + (1 - alpha) * fake_batch
                gradients = []
                epsilon = 1e-5

                for i in range(batch_size):
                    grad = np.zeros(interpolated.shape[1])
                    for j in range(interpolated.shape[1]):
                        delta = np.zeros(interpolated.shape[1])
                        delta[j] = epsilon
                        d_plus = self.discriminator_forward(interpolated[i:i + 1] + delta, condition[i:i + 1])
                        d_minus = self.discriminator_forward(interpolated[i:i + 1] - delta, condition[i:i + 1])
                        grad[j] = (d_plus - d_minus) / (2 * epsilon)
                    gradients.append(grad)

                penalties = [(np.linalg.norm(g) - 1) ** 2 for g in gradients]
                gp = self.gradient_penalty_weight * np.mean(penalties)

                d_loss = np.mean(self.discriminator_forward(fake_batch, condition)) - \
                         np.mean(self.discriminator_forward(real_batch, condition)) + gp

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = -np.mean(self.discriminator_forward(
                self.generator_forward(noise, real_labels),
                real_labels
            ))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

    def generate_samples(self, n_samples, class_label=1):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        labels = np.full((n_samples, 1), class_label)
        return self.generator_forward(noise, labels).astype(np.float32)