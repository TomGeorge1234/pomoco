from typing import Dict, Optional, Tuple, Any, Callable
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from synthetic_data.kernels import KERNELS

#TODO change this to a Matern 5/2 OU process for linear time complexity
#TODO write this in numpy or pytorch for easier downstream use

class GaussianProcess:
    """
    A class for generating samples from a multi-dimensional Gaussian Process (GP) using JAX.

    This class supports sampling from a GP with a specified kernel, generating samples
    at a coarse resolution first for efficiency, and then interpolating to a finer
    resolution.

    Attributes:
        dim (int): The dimensionality of the Gaussian Process (number of independent traces).
        kernel_type (str): The name of the kernel function to use (must be in KERNELS).
        kernel_params (Dict[str, Any]): Parameters passed to the kernel function (e.g., length scale).
        kernel_func (Callable): The actual kernel function retrieved from KERNELS.
    """

    def __init__(
        self, 
        dim: int, 
        kernel_type: str = "rbf", 
        kernel_params: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initializes the GaussianProcess generator.

        Args:
            dim: The number of independent dimensions to sample.
            kernel_type: The key for the kernel in the KERNELS dictionary. Defaults to "rbf".
            kernel_params: A dictionary of parameters for the kernel function. 
                           Defaults to an empty dict if None.

        Raises:
            ValueError: If the provided kernel_type is not found in KERNELS.
        """
        self.dim = dim
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params if kernel_params is not None else {}

        if self.kernel_type not in KERNELS:
            raise ValueError(
                f"Kernel type '{self.kernel_type}' not found. "
                f"Available kernels: {list(KERNELS.keys())}"
            )

        self.kernel_func = KERNELS[self.kernel_type]

    def sample(
        self, 
        T: float, 
        dt: float, 
        dt_sample: Optional[float] = None, 
        seed: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Samples trajectories from the Gaussian Process.

        This method first samples the GP at a coarser time resolution (`dt_sample`) 
        to reduce the computational cost of the covariance matrix decomposition, 
        and then linearly interpolates the result to the target resolution (`dt`).

        Args:
            T: The total duration of the simulation in seconds.
            dt: The target time step for the output array.
            dt_sample: The time step used for the internal covariance sampling. 
                       If None, defaults to `dt`. Larger values are faster but less accurate.
            seed: Random seed for reproducibility.

        Returns:
            A tuple containing:
                - time (jax.Array): Time vector of shape (num_steps,).
                - latent (jax.Array): Sampled GP trajectories of shape (num_steps, dim).
        """
        key = random.PRNGKey(seed)
        keys = random.split(key, self.dim)  # keys for sampling each dimension independently

        dt_sample = dt_sample or dt
        num_samples = int(T / dt_sample) + 1
        time_sample = jnp.linspace(0, T, num_samples)

        # Compute covariance matrix using the kernel function
        # Shape: (num_samples, num_samples)
        cov_matrix = self.kernel_func(time_sample, time_sample, self.kernel_params)
        cov_matrix += jnp.eye(num_samples) * 1e-4 # Jitter for stability
        mean = jnp.zeros(num_samples)

        def sample_one_dim(k: jax.Array) -> jax.Array:
            """Helper to sample one dimension given a PRNG key."""
            return random.multivariate_normal(k, mean, cov_matrix)

        # Vectorize sampling across the 'dim' independent dimensions
        # Output shape: (num_steps_coarse, dim)
        latent_process_samples = jax.vmap(sample_one_dim)(keys).T

        # Interpolate to the target time base
        num_steps = int(T / dt) + 1
        time = jnp.linspace(0, T, num_steps)

        def interp_one_dim(y: jax.Array) -> jax.Array:
            """Helper to interpolate one dimension to the target time base."""
            return jnp.interp(time, time_sample, y)

        # Vectorize interpolation across the dimension axis (axis 1)
        latent = jax.vmap(interp_one_dim, in_axes=1, out_axes=1)(latent_process_samples)

        # convert to numpy for downstream compatibility
        time = np.array(time)
        latent = np.array(latent)

        return time, latent


# Example usage
if __name__ == "__main__":
    gp_rbf = GaussianProcess(dim=3, kernel_type="rbf", kernel_params={"tau": 1.0})
    time, latent = gp_rbf.sample(T=10.0, dt=0.01, dt_sample=0.1, seed=42)
