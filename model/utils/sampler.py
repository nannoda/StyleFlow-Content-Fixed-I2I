import math
from typing import Optional
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class DistributedGivenIterationSampler(Sampler[int]):
    def __init__(self, 
                 dataset: np.ndarray,  # Assuming dataset is a numpy array (or can be a list)
                 total_iter: int,
                 batch_size: int,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 last_iter: int = -1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0


    def __iter__(self) -> iter:
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self) -> np.ndarray:
        """
        Generates a shuffled list of indices for the given dataset.
        Each process shuffles the list with the same seed and selects a piece according to the rank.
        :return: The list of indices to sample from
        """
        # Set the seed for reproducibility
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]
        assert len(indices) == self.total_size

        return indices


    def __len__(self) -> int:
        """
        Returns the total number of elements this sampler will iterate over.
        This method should not take into account the `last_iter` value as it is meant for display purposes.
        :return: The total size of the dataset for this sampler
        """
        return self.total_size
