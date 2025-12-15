import torch
from typing import Optional, List
from synthetic_data.generator import Generator
import numpy as np

class Dataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        data_generators: Generator,
        trial_length: float = 10.0,
    ):
        super().__init__()

        self.data_generators = data_generators
        self.trial_length = trial_length


    def __iter__(
        self,
    ):
        while True:
            data_dict = self.data_generators.sample(total_time=self.trial_length,)
            yield_dict = {
                'time': data_dict['time'],
                'latents': data_dict['latents'],
                'spike_counts': data_dict['spike_counts'],
            }
            yield yield_dict


if __name__ == "__main__":
    dataset = Dataset()
