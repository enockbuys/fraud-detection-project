import numpy as np
from scipy.stats import wasserstein_distance

def leaky_relu(x, alpha=0.2):#Stability in GAN training
    return np.maximum(alpha * x, x)
def leaky_relu_deriv(a, alpha=0.2):#Derivative used in backprop
    return np.where(a > 0, 1.0, alpha)
def tanh(x):
    return np.tanh(np.clip(x, -10, 10))
def tanh_deriv(x):
    t = tanh(x)
    return 1 - t ** 2

#for stable GAN training.
def initialize_layer(in_size, out_size):
    w = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size) * 0.01
    b = np.zeros(out_size)
    return w, b

def forward_pass(x, weights, is_critic=False):
    activations = [x]
    for w, b in weights[:-1]:
        z = np.dot(activations[-1], w) + b
        a = leaky_relu(z)
        activations.append(a)
    w, b = weights[-1]
    z = np.dot(activations[-1], w) + b
    output = z if is_critic else tanh(z)
    activations.append(output)
    return activations, output

#momentum + adaptive learning rate.
class AdamOptimizer:
    def __init__(self, shape, lr=0.0001, beta1=0.5, beta2=0.999, epsilon=1e-8):
        self.m = np.zeros(shape)#momentum
        self.v = np.zeros(shape)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1 #
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, w, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        updated = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return np.clip(updated, -1.0, 1.0)

def finite_diff_grad(x, weights, eps=1e-4):#Approximate numerical gradient
    grad = np.zeros_like(x)
    _, base_out = forward_pass(x, weights, is_critic=True)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[:, i] += eps
        _, out_plus = forward_pass(x_plus, weights, is_critic=True)
        diff = (out_plus - base_out) / eps
        grad[:, i] = diff.flatten()
    grad = np.clip(grad, -1.0, 1.0)
    return grad

def compute_gradient_penalty(real_data, fake_data, c_weights, lambda_gp=10.0):
    batch_size = real_data.shape[0]
    alpha = np.random.uniform(0.0, 1.0, size=(batch_size, 1))
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    _, critic_inter = forward_pass(interpolates, c_weights, is_critic=True)
    gradients = finite_diff_grad(interpolates, c_weights)
    grad_norm = np.sqrt(np.sum(gradients ** 2, axis=1) + 1e-8)
    penalty = lambda_gp * np.mean((grad_norm - 1.0) ** 2)#Lipschitz constraint
    return penalty

def backward_pass_delta(weights, delta, inputs, optim_w, optim_b, is_critic=False, is_generator=False):
    activations, output = forward_pass(inputs, weights, is_critic=is_critic)
    new_weights = []
    new_biases = []
    delta = delta.copy()
    for i in range(len(weights) - 1, -1, -1):
        w, b = weights[i]
        a = activations[i + 1] if i < len(weights) - 1 else output
        if i == len(weights) - 1 and is_critic:
            grad = delta
        else:
            grad = delta * leaky_relu_deriv(a)
        grad_w = np.dot(activations[i].T, grad) / inputs.shape[0]
        grad_b = np.mean(grad, axis=0)
        w_new = optim_w[i].update(w, grad_w)
        b_new = optim_b[i].update(b, grad_b)
        new_weights.insert(0, w_new)
        new_biases.insert(0, b_new)
        if i > 0:
            delta = np.dot(grad, w.T)
    return [(w, b) for w, b in zip(new_weights, new_biases)]

def backward_pass_real_fake(real_data, fake_data, c_weights, c_optim_w, c_optim_b, lambda_gp):#compute the critic’s predictions for real and fake batches
    _, c_real_out = forward_pass(real_data, c_weights, is_critic=True)
    _, c_fake_out = forward_pass(fake_data, c_weights, is_critic=True)
    batch_size = real_data.shape[0]
    delta_real = np.ones((batch_size, 1)) / batch_size
    delta_fake = -np.ones((batch_size, 1)) / batch_size
    wass_loss = np.mean(c_fake_out) - np.mean(c_real_out)#Wasserstein loss
    penalty = compute_gradient_penalty(real_data, fake_data, c_weights, lambda_gp)
    c_loss = wass_loss + penalty #critic loss
    delta = delta_fake
    c_weights = backward_pass_delta(c_weights, delta, fake_data, c_optim_w, c_optim_b, is_critic=True)
    return c_weights, c_loss

def initialize_network(feature_dim, latent_dim=128):
    g_weights, g_optim_w, g_optim_b = [], [], []
    c_weights, c_optim_w, c_optim_b = [], [], []
    layer_sizes = [latent_dim, 128, 64, feature_dim]
    for i in range(len(layer_sizes) - 1):
        w, b = initialize_layer(layer_sizes[i], layer_sizes[i + 1])
        g_weights.append((w, b))
        g_optim_w.append(AdamOptimizer(w.shape))
        g_optim_b.append(AdamOptimizer(b.shape))
    layer_sizes = [feature_dim, 64, 32, 1]
    for i in range(len(layer_sizes) - 1):
        w, b = initialize_layer(layer_sizes[i], layer_sizes[i + 1])
        c_weights.append((w, b))
        c_optim_w.append(AdamOptimizer(w.shape))
        c_optim_b.append(AdamOptimizer(b.shape))
    return g_weights, g_optim_w, g_optim_b, c_weights, c_optim_w, c_optim_b

def generate_synthetic_data(g_weights, n_samples, latent_dim, feature_dim):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    _, fake_data = forward_pass(noise, g_weights, is_critic=False)
    return fake_data

def evaluate_samples(real_data, fake_data):
    mean_diff = np.mean(np.abs(np.mean(real_data, axis=0) - np.mean(fake_data, axis=0)))
    std_diff = np.mean(np.abs(np.std(real_data, axis=0) - np.std(fake_data, axis=0)))
    kl_div = np.mean([wasserstein_distance(real_data[:, i], fake_data[:, i]) for i in range(real_data.shape[1])])
    return mean_diff, std_diff, kl_div

def train_gan(real_data, val_data=None, epochs=1000, batch_size=64, latent_dim=128, feature_dim=30,
              g_lr=0.0001, d_lr=0.0001, n_critic=5, lambda_gp=10.0, print_interval=100):
    print("Initializing network...")
    real_data = np.asarray(real_data)
    if val_data is not None:
        val_data = np.asarray(val_data)
    g_weights, g_optim_w, g_optim_b, c_weights, c_optim_w, c_optim_b = initialize_network(feature_dim, latent_dim)
    print("Network initialized")
    training_history = {'c_loss': [], 'g_loss': [], 'mean_diff': [], 'std_diff': [], 'kl_div': []}
    best_kl = float('inf')
    wait = 0
    patience = print_interval * 10
    n_samples = real_data.shape[0]

    for epoch in range(epochs):
        np.random.shuffle(real_data)
        total_c_loss = 0
        total_g_loss = 0
        total_penalty = 0
        num_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_size_actual = min(batch_size, n_samples - i)
            batch = real_data[i:i + batch_size_actual]

            for _ in range(n_critic):
                noise = np.random.normal(0, 1, (batch_size_actual, latent_dim))
                _, g_output = forward_pass(noise, g_weights, is_critic=False)
                c_weights, c_loss = backward_pass_real_fake(batch, g_output, c_weights, c_optim_w, c_optim_b, lambda_gp)
                total_c_loss += c_loss
                total_penalty += compute_gradient_penalty(batch, g_output, c_weights, lambda_gp)

            noise = np.random.normal(0, 1, (batch_size_actual, latent_dim))
            _, g_output = forward_pass(noise, g_weights, is_critic=False)
            _, c_fake_output = forward_pass(g_output, c_weights, is_critic=True)
            g_loss = -np.mean(c_fake_output)
            total_g_loss += g_loss

            delta_g = finite_diff_grad(g_output, c_weights)
            delta_g = -delta_g / batch_size_actual
            g_weights = backward_pass_delta(g_weights, delta_g, noise, g_optim_w, g_optim_b, is_critic=False, is_generator=True)

            num_batches += 1

        if num_batches == 0:
            continue

        if epoch % print_interval == 0:
            avg_c_loss = total_c_loss / (num_batches * n_critic)
            avg_g_loss = total_g_loss / num_batches
            avg_penalty = total_penalty / (num_batches * n_critic)

            test_samples = generate_synthetic_data(g_weights, min(50, n_samples), latent_dim, feature_dim)
            mean_diff, std_diff, kl_div = evaluate_samples(real_data, test_samples)
            training_history['c_loss'].append(avg_c_loss)
            training_history['g_loss'].append(avg_g_loss)
            training_history['mean_diff'].append(mean_diff)
            training_history['std_diff'].append(std_diff)
            training_history['kl_div'].append(kl_div)

            val_kl = kl_div
            if val_data is not None:
                val_mean_diff, val_std_diff, val_kl = evaluate_samples(val_data, test_samples)

            if epoch % print_interval == 0:
                print(f"Epoch {epoch}: C Loss: {avg_c_loss:.4f}, G Loss: {avg_g_loss:.4f}, Val KL: {val_kl:.4f}")

            if val_data is not None:
                if val_kl < best_kl:
                    best_kl = val_kl
                    wait = 0
                else:
                    wait += print_interval
                    if wait >= patience or avg_penalty > 1000:
                        print(f"Early stopping at epoch {epoch}")
                        break

    final_losses = {'g_loss': avg_g_loss, 'd_loss': avg_c_loss}
    return g_weights, training_history, final_losses