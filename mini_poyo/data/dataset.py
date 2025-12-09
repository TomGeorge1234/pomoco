import torch
from typing import Optional
from synthetic_data.generator import generator
from synthetic_data.gaussian_process import GaussianProcess
import numpy as np

class Dataset(torch.utils.data.IterableDataset):
    """Initialises a dataset from a config file """
    def __init__(self):
        super().__init__()

        # if isinstance(config, str):
        #     with open(config, 'r') as f:
        #         config_dict = yaml.safe_load(f)
        #     self.config = OmegaConf.create(config_dict)

        self.latent_timescale = 1.0
        latent_dim = 2

        num_sets = 2
        num_neurons = 100
        self.max_firing_rate = 50
        self.trial_duration = 2.0

        self.gaussian_process = GaussianProcess(
            dim=latent_dim, 
            kernel_type="rbf", 
            kernel_params={"tau": self.latent_timescale}
        )

        self.dataset_weight_dict = {i : self.get_weight_matrix(num_neurons, latent_dim) for i in range(num_sets)}
        
    def __iter__(self):
        while True: 
            # randomly select a dataset
            dataset_idx = torch.randint(0, len(self.dataset_weight_dict), (1,)).item()
            weight_matrix = self.dataset_weight_dict[dataset_idx]

            # generate data 
            spike_times, spike_unit_ids, time, latents = generator(
                weight_matrix=weight_matrix.numpy(),
                gaussian_process=self.gaussian_process,
                T=self.trial_duration,
                dt=0.01,
                max_fr=self.max_firing_rate,
            )
            yield {
                "spike_times": torch.from_numpy(spike_times).float(),
                "spike_unit_ids": torch.from_numpy(spike_unit_ids).long(),
                "time": torch.from_numpy(time).float(),
                "latents": torch.from_numpy(latents).float(),
            }

    
    def get_weight_matrix(self, num_neurons, latent_dim):
        weight_matrix = torch.randn(num_neurons, latent_dim)
        weight_matrix = weight_matrix / np.sqrt(latent_dim)
        return weight_matrix




        

if __name__ == "__main__":
    dataset = Dataset()