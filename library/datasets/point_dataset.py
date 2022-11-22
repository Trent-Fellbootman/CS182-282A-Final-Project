from typing import List, Tuple, Callable
import jax
from jax import numpy as jnp, random

from base import Dataset, Distribution
from utils import TensorDataset, PolynomialPointSampler


class PointPairDataset(Distribution):
    """Point dataset.
    """
    
    def __init__(self, inputs: jnp.ndarray,
                 sampler_A: Callable, sampler_B,
                 noise_scale_A: float=0.1, noise_scale_B: float=0.1,
                 key: random.KeyArray = random.PRNGKey(0)):
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
        super().__init__()
        
        self.__manifold_dim = inputs.shape[-1]
        
        self.__sampler_A = sampler_A
        self.__sampler_B = sampler_B
        self.__noise_scale_A = noise_scale_A
        self.__noise_scale_B = noise_scale_B
        
        self.__random_state = key
        
        # points_A = sampler_A(inputs)
        # points_B = sampler_B(inputs)
        
        # key_1, key_2 = random.split(key)
        
        # noise_A, noise_B = noise_scale_A * random.normal(key_1, points_A.shape), noise_scale_B * random.normal(key_2, points_B.shape)
        
        # points_A = points_A + noise_A
        # points_B = points_B + noise_B
        
        # super().__init__(points_A, points_B)
    
    @property
    def manifold_dim(self):
        return self.__manifold_dim
    
    # def get_point_pairs(self):
    #     return self.get_tensors()
    
    @staticmethod
    def generate_random_distribution_by_taylor(manifold_dim: int, dis_A_dim: int, dis_B_dim: int, manifold_range: float,
                                          max_order: int, coeff_range: float, noise_std_A: float, noise_std_B: float,
                                          key: random.KeyArray=random.PRNGKey(0)):
        """Generate random dataset by mapping randomly-generated points through randomly-generated polynomials.

        Args:
            manifold_dim (int): The number of manifold dimensions.
            dis_A_dim (int): Ambient dimensionality for distribution A.
            dis_B_dim (int): Ambient dimensionality for distribution B.
            manifold_range (float): The points will be distributed uniformly in `[-manifold_range, manifold_range) ** manifold_dim`.
            before being transformed by polynomials.
            max_order (int): The maximum order of polynomial transformations.
            coeff_range (float): Coefficients of polynomials will be distributed uniformly in `[-coeff_range, coeff_range)`.
            noise_std_A (float): The noise to apply to the points in distribution A after transformation.
            noise_std_B (float): The noise to apply to the points in distribution B after transformation.
        """
        
        key_manifold, key_sampler_A, key_sampler_B, key_noise_A, key_noise_B = random.split(key, num=5)
        inputs = random.normal(key_manifold, (n_samples, manifold_dim))
        sampler_A = PolynomialPointSampler.generateRandomSampler(manifold_dim, dis_A_dim, max_order, coeff_range, key_sampler_A)
        sampler_B = PolynomialPointSampler.generateRandomSampler(manifold_dim, dis_B_dim, max_order, coeff_range, key_sampler_B)

