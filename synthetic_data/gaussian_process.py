from typing import Dict, Optional, Tuple
import warnings
import numpy as np
from synthetic_data.generator import Generator

class GPSpikeGenerator(Generator):
    """
    A generator class for producing spike count data from a Gaussian Process (GP) latent model.
    """
    def __init__(
        self,
        # GP parameters 
        latent_dim: int,
        latent_timescale: float,
        kernel_type: str = "rbf",
        kernel_params: Optional[Dict[str, float]] = None,
        dt: float = 0.1,
        # Neuron parameters 
        median_firing_rate: float = 20.0,
        num_neurons: int = 100,
        weights = None,
        ) -> None:
        
        super().__init__()
        self.gp = GaussianProcess(latent_dim, kernel_type, kernel_params)
        if weights is not None: 
            assert weights.shape == (latent_dim, num_neurons)
            self.weights = weights
        else: 
            self.weights = np.random.randn(latent_dim, num_neurons)
        self.median_firing_rate = median_firing_rate
        self.num_neurons = num_neurons
        self.dt = dt
        if dt > latent_timescale / 5: 
            warnings.warn("dt is larger compared to the latent timescale.")
        self.dt_sample = max(latent_timescale / 10.0,dt)  # Sample at a finer resolution than the timescale

    def sample(
            self,
            total_time: float,
            ) -> Tuple[np.ndarray, np.ndarray]:

        # Sample latent from GP
        time, latents = self.gp.sample(
            total_time=total_time,
            dt=self.dt,
            dt_sample=self.dt_sample, 
            seed = None,)

        # Compute firing rates: z_i ~ N(0,1), W_ij ~ N(0,1) => logrates ~ N(0, num_latents)
        logrates =  latents @ self.weights  # ~ N(0, latent_dim)
        logrates = logrates / np.sqrt(self.gp.dim)  # ~ N(0,1)
        print(f"Checking distribution of normalized logrates: mean={logrates.mean():.2f}, std={logrates.std():.2f}")
        rates = np.exp(0.3 * logrates) # ~ lnN(0,1) 
        rates = self.median_firing_rate * rates # ~ lnN(ln(median_fr),1)
        print(f"Checking distribution of lognormalize rates: median={np.median(rates):.2f}, std={rates.std():.2f}")

        # Sample spike counts and times
        lam = rates * self.dt  # shape: (num_timepoints, num_neurons)
        spike_counts = np.random.poisson(lam)  # shape: (num_timepoints, num_neurons)

        # Extract sample individual spikes; return times and unit ids
        t_ids, n_ids = np.nonzero(spike_counts)
        counts_for_nonzero_bins = spike_counts[t_ids, n_ids]

        spike_unit_ids = np.repeat(n_ids, counts_for_nonzero_bins)
        base_spike_times = np.repeat(time[t_ids], counts_for_nonzero_bins)
        spike_times = base_spike_times + np.random.uniform(0, self.dt, size=base_spike_times.shape)
        return time, latents, spike_counts, spike_times, spike_unit_ids


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
        total_time: float,  # what's a better name for this parameter...
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
            total_time: The total duration of the simulation in seconds.
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
        num_samples = int(total_time / dt_sample) + 1
        time_sample = np.linspace(0, total_time, num_samples)

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
        num_steps = int(total_time / dt) + 1
        time = np.linspace(0, total_time, num_steps)
        
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