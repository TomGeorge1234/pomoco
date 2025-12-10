import torch
from typing import Optional
from synthetic_data.generator import generator
import numpy as np


class Dataset(torch.utils.data.IterableDataset):
    """Initialises a dataset from a config file"""

    def __init__(
        self,
        latent_dim=2,
        latent_timescale=1.0,
        num_sets=1,
        num_neurons=100,
        max_firing_rate=50,
        trial_duration=2.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_timescale = latent_timescale
        self.num_sets = num_sets
        self.num_neurons = num_neurons
        self.max_firing_rate = max_firing_rate
        self.trial_duration = trial_duration

        self.dataset_weight_dict = {
            i: self.get_weight_matrix(self.num_neurons, self.latent_dim)
            for i in range(self.num_sets)
        }

    def __iter__(
        self,
    ):
        while True:
            # randomly select a dataset
            dataset_idx = np.random.randint(0, self.num_sets)
            weight_matrix = self.dataset_weight_dict[dataset_idx]

            # generate data
            time, latents, spike_counts = generator(
                weight_matrix=weight_matrix,
                latent_dim=self.latent_dim,
                latent_timescale=self.latent_timescale,
                trial_duration=self.trial_duration,
                dt=0.01,
                max_fr=self.max_firing_rate,
            )
            yield {
                "time": torch.from_numpy(time).float(),
                "latents": torch.from_numpy(latents).float(),
                "spike_counts": torch.from_numpy(spike_counts).float(),
            }

    def get_weight_matrix(self, num_neurons, latent_dim):
        weight_matrix = np.random.randn(num_neurons, latent_dim)
        weight_matrix = weight_matrix / np.sqrt(latent_dim)
        return weight_matrix


if __name__ == "__main__":
    dataset = Dataset()
