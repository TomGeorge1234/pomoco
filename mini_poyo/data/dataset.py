import torch
# import config using yaml and omegaconf
import yaml
from omegaconf import OmegaConf
from typing import Optional

class Dataset(torch.utils.data.IterableDataset):
    """Initialises a dataset from a config file """
    def __init__(
        self, 
        config: Optional[str] = None, #path to config file
        split: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = OmegaConf.create(config_dict)


        self.split = split
        latent_timescale = self.config.data.latent_timescale
        latent_dim = self.config.data.latent_dim

        self.num_sets = self.config.data.num_sets
        self.num_neurons = self.config.data.num_neurons
        self.max_firing_rate = self.config.data.max_firing_rate
        self.trial_duration = self.config.data.trial_duration

        self.


        

if __name__ == "__main__":
    dataset = Dataset(config="configs/synthetic_data.yaml", split="train")