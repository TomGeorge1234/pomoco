from typing import Dict, Optional, Tuple
import numpy as np
from generator import Generator

class GaussianProcessSpikeCount(Generator):
    """
    A generator class for producing spike count data from a Gaussian Process (GP) latent model.
    """
    def __init__(
        self,
        latent_dim: int,
        kernel_type: str = "rbf",
        kernel_params: Optional[Dict[str, float]] = None,
        max_firing_rate: float = 50.0,
        non_linearity: str = "exponential",
        num_neurons: int = 100,
        weights = None,
    ) -> None:
        
        super().__init__()

        self.gp = GaussianProcess(latent_dim, kernel_type, kernel_params)
        
        if weights is not None: 
            assert weights.shape = (latent_dim, num_neurons)

        self.max_firing_rate = max_firing_rate
        self.non_linearity = non_linearity
        self.num_neurons = num_neurons
        if weights is None:
            self.weights = self.create_weight_matrix(num_neurons, dim)
        else:
            self.weights = weights

    def 

class GaussianProcess:
    """
    A class for generating samples from a multi-dimensional Gaussian Process (GP) using NumPy.

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples trajectories from the Gaussian Process.

        This method first samples the GP at a coarser time resolution (`dt_sample`) 
        to reduce the computational cost then linearly interpolates the result to the target resolution (`dt`).
        Note: this dt_sample should be smaller than the length scale of the kernel for accurate results.

        Args:
            T: The total duration of the simulation in seconds.
            dt: The target time step for the output array.
            dt_sample: The time step used for the internal covariance sampling. 
                       If None, defaults to `dt`. Larger values are faster but less accurate.
            seed: Random seed for reproducibility.

        Returns:
            A tuple containing:
                - time (np.ndarray): Time vector of shape (num_steps,).
                - latent (np.ndarray): Sampled GP trajectories of shape (num_steps, dim).
        """
        # Initialize NumPy random generator
        rng = np.random.default_rng(seed)

        dt_sample = dt_sample or dt
        num_samples = int(T / dt_sample) + 1
        time_sample = np.linspace(0, T, num_samples)

        # Compute covariance matrix using the kernel function
        # Shape: (num_samples, num_samples)
        cov_matrix = self.kernel_func(time_sample, time_sample, self.kernel_params)
        cov_matrix += np.eye(num_samples) * 1e-4  # Jitter for stability
        mean = np.zeros(num_samples)

        # Sample from the multivariate normal distribution
        # We can ask for `dim` samples at once using the `size` argument.
        # Output of multivariate_normal with size=dim is (dim, num_samples).
        # We transpose to get (num_samples, dim) to match time on axis 0.
        latent_process_samples = rng.multivariate_normal(
            mean, cov_matrix, size=self.dim
        ).T

        # Interpolate to the target time base
        num_steps = int(T / dt) + 1
        time = np.linspace(0, T, num_steps)
        
        # Initialize output array
        latent = np.zeros((num_steps, self.dim))

        # NumPy doesn't have a vmap equivalent for interp, so we loop over dimensions.
        # This is generally fast enough for typical latent dimensions (e.g. <1000).
        for i in range(self.dim):
            latent[:, i] = np.interp(time, time_sample, latent_process_samples[:, i])

        return time, latent
    

def rbf_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    # Broadcasting: (N, 1) - (1, M) -> (N, M)
    diff = (x1[:, None] - x2[None, :]) / tau
    return np.exp(-0.5 * diff**2)

def matern_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    nu = params.get("nu", 1.5)
    dist = np.abs(x1[:, None] - x2[None, :]) / tau
    
    if nu == 0.5:
        return np.exp(-dist)
    elif nu == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        return (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
    elif nu == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        return (1 + sqrt5_dist + 5 / 3 * dist**2) * np.exp(-sqrt5_dist)
    else:
        raise NotImplementedError(
            "Only nu=0.5, 1.5, 2.5 are implemented for Matern kernel"
        )

def periodic_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    period = params.get("period", 1.0)
    dist = np.abs(x1[:, None] - x2[None, :])
    return np.exp(-2 * (np.sin(np.pi * dist / period) / tau) ** 2)

KERNELS = {"rbf": rbf_kernel, "matern": matern_kernel, "periodic": periodic_kernel}





# Example usage
if __name__ == "__main__":
    gp_rbf = GaussianProcess(dim=3, kernel_type="rbf", kernel_params={"tau": 1.0})
    time, latent = gp_rbf.sample(T=10.0, dt=0.01, dt_sample=0.1, seed=42)
    print(f"Time shape: {time.shape}")
    print(f"Latent shape: {latent.shape}")