import jax
from jax import numpy as jnp, random
from jax import tree_util
from typing import List, Iterable
from tqdm import tqdm

"""This file implements the dataset base class.
"""


class Dataset:
    """Base Dataset class.

    Here, each sample is assumed to be a pytree (i.e., recursive structure consisting of tuples, lists and dicts.)

    Behavior is UNDEFINED if sample type is not pytree.

    To implement your own dataset, you must override 3 methods:

    - `__init__(self)`

        Constructor.

    - `__getitem__(self, i: int) -> Any`

        Gets the sample at index `i`. If `i` is outside of `[-len(self), len(self))`, return None.

    - `__len__(self) -> int`

        returns the length of this dataset.
    """

    def __init__(self):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        pass

    # def buildBatch(self, indices: Iterable[int]):
    #     """Build a batch.

    #     Args:
    #         `indices` (Iterable[int]): Indices of samples to take.

    #     Returns:
    #         A pytree where each leaf contains a batch that combines the corresponding leaves in all samples indicated by `indices`.
    #     """
    #     # assert -self.__len__() <= min(indices) <= max(indices) < self.__len__(), "Found invalid indices!"

    #     return tree_util.tree_map(lambda *args: jnp.array(args),
    #                               *[self.__getitem__(index) for index in indices])


class DataLoader:
    """Base DataLoader class. Automatically handles batching and shuffling.

    An object of this class is STATEFUL.

    You should not need to inherit from this class directly.
    
    ## Note
    - Prefetch is performed automatically in constructor.
    """

    def __init__(self, dataset: Dataset, batch_size: int, key: random.KeyArray = random.PRNGKey(0), auto_reshuffle: bool = True):
        """Constructor.

        Args:
            dataset (Dataset): dataset where samples are stored.
            batch_size (int): batch size.
            key (random.KeyArray): initial key to be used in psuedo random number generator (PRNG).
            auto_reshuffle (bool): whether or not to reshuffle the dataset after each iteration.
        """

        self.__key = key
        self.__batch_size = batch_size
        self.__auto_reshuffle = auto_reshuffle

        self.__current_index = 0
        self.__max_index = len(dataset) // self.__batch_size - 1

        self.__samples = [dataset[index] for index in range(len(dataset))]
        
        # to be filled
        self.__batches = []
        self.reshuffle()
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def auto_reshuffle(self):
        return self.__auto_reshuffle

    def __init_batches(self):
        start_indices = list(range(0, len(self.__samples) - self.__batch_size + 1,
                                   self.__batch_size))

        self.__batches = [tree_util.tree_map(
            lambda *args: jnp.array(args),
            *self.__samples[index: index + self.__batch_size])
            for index in start_indices]

    def reshuffle(self, key: random.KeyArray | None = None):
        """Reshuffles and re-batches the data loader IN PLACE.
        
        Batches are regenerated.

        You should NOT call this method directly.

        Args:
            key (random.KeyArray | None, optional): PRNG key to be used. If None, use the key in current state and update current state.
        """
        
        if self.__current_index != 0:
            raise Exception("This dataloader is currently in iteration!")

        if key is None:
            self.__key, key = random.split(self.__key)
        else:
            key = key
        
        permutation = random.permutation(key, len(self.__samples))

        self.__samples = [self.__samples[index] for index in permutation]
        self.__init_batches()
    
    def __iter__(self):
        if self.__current_index != 0:
            raise Exception("This dataloader is currently in iteration!")
        
        return self

    def __next__(self):
        if self.__current_index > self.__max_index:
            self.__current_index = 0
            
            if self.__auto_reshuffle:
                self.reshuffle()
            
            raise StopIteration
        else:
            ret = self.__batches[self.__current_index]
            self.__current_index += 1

            return ret
    
    def restart_iteration(self):
        """Restart iteration. No reshuffling / rebatching happens.
        """
        
        self.__current_index = 0


# class Dummy(Dataset):

#     def __init__(self, num: int):
#         super().__init__()
#         self.upperbound = num

#     def __getitem__(self, index: int):
#         assert -self.upperbound <= index < self.upperbound, "out of bound!"
#         val = self.upperbound + index if index < 0 else index
#         return (val, (val ** 2, (val ** 3,)))

#     def __len__(self):
#         return self.upperbound
