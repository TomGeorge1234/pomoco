import torch
from typing import Optional, List
from synthetic_data.generator import Generator
import numpy as np

class Dataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        data_generators: List[Generator],
    ):
        super().__init__()

        self.num_sets = len(data_generators)
        self.data_generators = data_generators

        #TODO create a dict storing info for each dataset and a hash to identify them


    def __iter__(
        self,
    ):
        while True:
            # randomly select a dataset
            dataset_idx = np.random.randint(0, self.num_sets)
            generator = self.data_generators[dataset_idx]

            time, latents, spike_counts = generator.generate_data()
            yield {
                "time": torch.from_numpy(time).float(),
                "latents": torch.from_numpy(latents).float(),
                "spike_counts": torch.from_numpy(spike_counts).float(),
                "dataset_idx": dataset_idx,
            }




if __name__ == "__main__":
    dataset = Dataset()
