from typing import Callable
from functools import partial
import jax
from jax import numpy as jnp, random, tree_util
from optax._src.base import PyTree
from flax import serialization
import pickle


class NumericalCheckingRecord:
    
    def __init__(self, data: PyTree):
        """Constructor.

        Args:
            data (PyTree): The data arrays for this object to hold.
        """
        
        self.__data = data
    
    @property
    def data(self):
        return self.__data
    
    def save(self, filepath: str):
        """Save the numeric records.

        Args:
            filepath (str): The filepath to save the records to.
        """
        
        with open(filepath, 'xb') as f:
            f.write(pickle.dumps(self.__data))
    
    def check(self, data: PyTree, check_fn: Callable=jnp.allclose):
        """Check the data against the record.

        Args:
            data (PyTree): The incoming data arrays to check.
            
            check_fn (Callable): The function to apply for each record-incoming data pair.
            The signature of this function should be (stored: jnp.ndarray, incoming: jnp.ndarray) -> bool.
            This function should take in two jnp.ndarrays and return a bool value, indicating
            whether the test was successful.
        
        Returns:
            A bool indicating if the test was successful for all stored-incoming pairs.
        """
        
        test_result = True
        
        def check(x, y):
            nonlocal test_result
            test_result = (test_result and check_fn(x, y))
        
        tree_util.tree_map(check, self.__data, data)
        
        return test_result
    
    @staticmethod
    def load(filepath: str):
        """Load a checker object.

        Args:
            filepath (str): The file to load.
        """
        
        with open(filepath, 'rb') as f:
            data = pickle.loads(f.read())
        
        return NumericalCheckingRecord(data)