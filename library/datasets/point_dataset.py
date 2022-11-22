from typing import List, Tuple, Callable
import jax
from jax import numpy as jnp, random

from base import Dataset
from utils import TensorDataset, Polynomial

class PointPairDataset(TensorDataset):
    """Point dataset.
    """
    
    def __init__(self, inputs: jnp.ndarray, sampler_A: Callable, sampler_B, noise_scale: float=0.1, key: random.KeyArray = random.PRNGKey(0)):
        """Constructor.

        Args:
            inputs (jnp.ndarray): The points in the input space that are passed to the samplers.
            
            sampler_A (Callable): The first sampler to generate data points in one distribution.
            The sampler must take in an n-d array and batch-evaluate the values. Using a high-order
            polynomial is a good idea, since polynomials can basically approximate any function
            according to Taylor's Theorem.
            
            sampler_B (Callable): The second sampler to generate data points in the other distribution.
            
            noise_scale (float): The scale of the Gaussian noise applied after transforming the inputs.
            
            key (random.KeyArray): The PRNG key to use.
        """
        
        points_A = sampler_A(inputs)
        points_B = sampler_B(inputs)
        
        key_1, key_2 = random.split(key)
        
        noise_A, noise_B = noise_scale * random.normal(key_1, points_A.shape), noise_scale * random.normal(key_2, points_B.shape)
        
        points_A = points_A + noise_A
        points_B = points_B + noise_B
        
        super().__init__(points_A, points_B)
