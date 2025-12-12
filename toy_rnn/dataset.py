import torch
from typing import Optional, List
from synthetic_data.generator import Generator
import numpy as np

class Dataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        data_generators: List[Generator],
        trial_length: float = 10.0,
    ):
        super().__init__()

        self.num_sets = len(data_generators)
        self.data_generators = data_generators
        self.trial_length = trial_length

        #TODO create a dict storing info for each dataset and a hash to identify them


    def __iter__(
        self,
    ):
        while True:
            # randomly select a dataset
            dataset_idx = np.random.randint(0, self.num_sets)
            generator = self.data_generators[dataset_idx]

            data_dict = generator.sample(
                total_time=self.trial_length,
            )
            time, latents, spike_counts = data_dict['time'], data_dict['latents'], data_dict['spike_counts']

            yield_dict = {
                'time': time,
                'latents': latents,
                'spike_counts': spike_counts,
                'dataset_idx': dataset_idx,
            }

            yield yield_dict




if __name__ == "__main__":
    dataset = Dataset()
