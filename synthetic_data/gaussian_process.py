import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from kernels import KERNELS


class GaussianProcess:
    def __init__(self, dim, kernel_type="rbf", kernel_params=None):
        self.dim = dim
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params if kernel_params is not None else {}

        if self.kernel_type not in KERNELS:
            raise ValueError(
                f"Kernel type '{self.kernel_type}' not found. Available kernels: {list(KERNELS.keys())}"
            )

        self.kernel_func = KERNELS[self.kernel_type]

    def sample(self, T, dt, dt_sample=None, seed=0):
        key = random.PRNGKey(seed)
        keys = random.split(key, self.dim) # for sampling each dimension
        dt_sample = dt_sample or dt
        num_samples = int(T / dt_sample) + 1
        time_sample = jnp.linspace(0, T, num_samples)

        # Compute covariance matrix and sample latent process
        cov_matrix = self.kernel_func(time_sample, time_sample, self.kernel_params)
        cov_matrix += jnp.eye(num_samples) * 1e-6 # for numerical stability
        mean = jnp.zeros(num_samples)
        def sample_one_dim(k):
            return random.multivariate_normal(k, mean, cov_matrix)
        latent_process_samples = jax.vmap(sample_one_dim)(keys).T  # (num_steps, dim)

        # Interpolate to target dt
        num_steps = int(T / dt) + 1
        time = jnp.linspace(0, T, num_steps)
        def interp_one_dim(y):
            return jnp.interp(time, time_sample, y)
        latent = jax.vmap(interp_one_dim, in_axes=1, out_axes=1)(latent_process_samples)

        return time, latent

    def plot(self, time, latent):
        plt.figure(figsize=(10, 6))
        for i in range(self.dim):
            plt.plot(time, latent[:, i], label=f"Dimension {i + 1}")
        plt.title(f"Sampled Gaussian Process Time Series ({self.kernel_type} kernel)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example 1: RBF Kernel
    print("Sampling with RBF Kernel...")
    gp_rbf = GaussianProcess(dim=3, kernel_type="rbf", kernel_params={"tau": 1.0})
    time, latent = gp_rbf.sample(T=10.0, dt=0.01, dt_sample=0.1, seed=42)
    gp_rbf.plot(time, latent)