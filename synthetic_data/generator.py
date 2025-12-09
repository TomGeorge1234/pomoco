from gaussian_process import GaussianProcess
import numpy as np


def generator(
    weight_matrix,
    gaussian_process: GaussianProcess,
    T=2.0,
    dt=0.1,
    max_fr=100.0,
):
    # Sample latent variables from Gaussian Process
    time, latents = gaussian_process.sample(T=T, dt=dt)
    print(np.std(latents))
    # Map to firing rates
    log_firing_rates = latents @ weight_matrix.T  # ReLU activation
    rates = np.exp(log_firing_rates)
    # Scale so max_fr = 3std and saturate using tanh
    rates = np.tanh(rates / 3) * max_fr

    # Sample spike counts and times
    lam = rates * dt  # shape: (num_timepoints, num_neurons)
    print(lam)
    spike_counts = np.random.poisson(lam)  # shape: (num_timepoints, num_neurons)
    print(f"Spike counts shape is: {spike_counts.shape}")

    # Extract individual spike instances 
    t_ids, n_ids = np.nonzero(spike_counts)
    counts_for_nonzero_bins = spike_counts[t_ids, n_ids]

    spike_unit_ids = np.repeat(n_ids, counts_for_nonzero_bins)
    base_spike_times = np.repeat(time[t_ids], counts_for_nonzero_bins)
    spike_times = base_spike_times + np.random.uniform(0, dt, size=base_spike_times.shape)

    return spike_times, spike_unit_ids, time, latents


if __name__ == "__main__":
    gp = GaussianProcess(dim=5, kernel_type="rbf", kernel_params={"tau": 1.0})
    gp.sample(T=5.0, dt=0.01, seed=0)
    W = np.random.randn(50, 5)  # 5 neurons, 2 latent dimensions
    spike_times, spike_unit_ids, time, latents = generator(
        weight_matrix=W,
        gaussian_process=gp,
        T=5.0,
        dt=0.01,
        max_fr=5.0,
    )
    print(f"Generated {len(spike_times)} spikes.")