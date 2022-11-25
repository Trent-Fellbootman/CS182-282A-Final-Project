from .base import Dataset, Distribution
from .utils import TensorDataset, PolynomialTransformation, PointDistribution

from typing import List, Tuple, Callable
import jax
from jax import numpy as jnp, random

import plotly.express as px
from plotly import subplots, graph_objects as go

import pandas as pd


class PointPairDistribution(Distribution):
    """Point dataset.
    """

    def __init__(self, latent_distribution: PointDistribution,
                 transformation_A: Callable, transformation_B,
                 noise_scale_A: float = 0.1, noise_scale_B: float = 0.1,
                 key: random.KeyArray = random.PRNGKey(0)):
        """Constructor.

        Args:
            inputs (jnp.ndarray): The points in the input space that are passed to the transformations.

            transformation_A (Callable): The first transformation to generate data points in one distribution.
            The transformation must take in an n-d array and return the batch-evaluated values. Using a high-order
            polynomial is a good idea, since polynomials can basically approximate any function
            according to Taylor's Theorem.

            transformation_B (Callable): The second transformation to generate data points in the other distribution.

            noise_scale (float): The scale of the Gaussian noise applied after transforming the inputs.

            key (random.KeyArray): The PRNG key to use.
        """
        super().__init__()

        self.__latent_distribution = latent_distribution

        self.__transformation_A = transformation_A
        self.__transformation_B = transformation_B
        self.__noise_scale_A = noise_scale_A
        self.__noise_scale_B = noise_scale_B

        self.__random_state = key

        # points_A = transformation_A(inputs)
        # points_B = transformation_B(inputs)

        # key_1, key_2 = random.split(key)

        # noise_A, noise_B = noise_scale_A * random.normal(key_1, points_A.shape), noise_scale_B * random.normal(key_2, points_B.shape)

        # points_A = points_A + noise_A
        # points_B = points_B + noise_B

        # super().__init__(points_A, points_B)

    @property
    def manifold_dim(self):
        return self.__latent_distribution.n_dims

    def draw_samples(self, n_samples: int, key: random.KeyArray = None):
        """Batch-draw samples.

        Args:
            n_samples (int): The number of samples to draw.
            key (random.KeyArray): The random key to use. If None, use self.__random_state and update the state.
            If not None, random state will NOT be updated.

        Returns:
            jnp.ndarray: Generated samples. Last dimension is feature dimension.
        """

        if key is None:
            self.__random_state, key = random.split(self.__random_state)

        key_inputs, key_A, key_B = random.split(key, num=3)

        latent_inputs = self.__latent_distribution.draw_samples(
            n_samples, key_inputs)

        points_A = self.__transformation_A(latent_inputs)
        points_B = self.__transformation_B(latent_inputs)
        points_A = points_A + self.__noise_scale_A * \
            random.normal(key_A, points_A.shape)
        points_B = points_B + self.__noise_scale_B * \
            random.normal(key_B, points_B.shape)

        return (latent_inputs, points_A, points_B)

    # def get_point_pairs(self):
    #     return self.get_tensors()

    @staticmethod
    def random_taylor(latent_dim: int, dis_A_dim: int, dis_B_dim: int, latent_range: float,
                      max_order: int, coeff_range: float, noise_std_A: float = 0.1, noise_std_B: float = 0.1,
                      key: random.KeyArray = random.PRNGKey(0)):
        """Generate random distribution by mapping randomly-generated points through randomly-generated polynomials.

        Args:
            latent_dim (int): The number of manifold dimensions.

            dis_A_dim (int): Ambient dimensionality for distribution A.

            dis_B_dim (int): Ambient dimensionality for distribution B.

            latent_range (float): The points will be distributed uniformly in `[-latent_range, latent_range)`.
            before being transformed by polynomials.

            max_order (int): The maximum order of polynomial transformations.

            coeff_range (float): Coefficients of polynomials will be distributed uniformly in `[-coeff_range, coeff_range)`.

            noise_std_A (float): The std of the Gaussian noise to apply to the points in distribution A after transformation.

            noise_std_B (float): The std of the Gaussian noise to apply to the points in distribution B after transformation.
        """

        key_manifold, key_transformation_A, key_transformation_B, key_pair = random.split(
            key, num=4)

        latent_dis = PointDistribution(n_dims=latent_dim,
                                       sampler=(
                                           random.uniform,
                                           {'minval': -latent_range,
                                               'maxval': latent_range}
                                       ),
                                       key=key_manifold)

        transformation_A = PolynomialTransformation.generate_random(
            latent_dim, dis_A_dim, max_order, coeff_range, key_transformation_A)
        transformation_B = PolynomialTransformation.generate_random(
            latent_dim, dis_B_dim, max_order, coeff_range, key_transformation_B)

        return PointPairDistribution(latent_dis, transformation_A, transformation_B, noise_std_A, noise_std_B, key_pair)

    def __str__(self):
        lines = ['PointPairDataset:']
        lines.append(f'latent distribution:')
        lines += ['\t' +
                  line for line in str(self.__latent_distribution).split('\n')]
        lines.append('transformation A:')
        lines += ['\t' +
                  line for line in str(self.__transformation_A).split('\n')]
        lines.append('transformation B:')
        lines += ['\t' +
                  line for line in str(self.__transformation_B).split('\n')]
        lines.append(f'Noise std A: {self.__noise_scale_A:.2e}')
        lines.append(f'Noise std B: {self.__noise_scale_B:.2e}')

        return '\n'.join(lines)

    def generate_dataset(self, n_samples: int, key: random.KeyArray = random.PRNGKey(0)):
        """Generate a dataset from this distribution.

        Args:
            n_samples (int): Number of samples.
            key (random.KeyArray, optional): Random key to use.

        Returns:
            PointPairDataset: Generated dataset.
        """

        latent_inputs, points_A, points_B = self.draw_samples(n_samples, key)

        return PointPairDataset(latent_inputs, points_A, points_B)


class PointPairDataset(Dataset):

    """Point pair dataset.
    """

    def __init__(self, latent_points: jnp.ndarray, points_A: jnp.ndarray, points_B: jnp.ndarray):
        """Constructor.

        Args:
            latent_points (jnp.ndarray): "Input" points in latent space.
            points_A (jnp.ndarray): Noisy transformed points in ambient space A.
            points_B (jnp.ndarray): Noisy transformed points in ambient space B.
        """

        super().__init__()

        assert latent_points.shape[0] == points_A.shape[0] == points_B.shape[0]

        self.__latent_points = latent_points
        self.__points_A = points_A
        self.__points_B = points_B

    def __getitem__(self, index: int):
        return (self.__points_A[index], self.__points_B[index])

    def __len__(self):
        return self.__latent_points.shape[0]

    def create_visualization(self):
        """Creates a visualization of this dataset.

        Returns:
            fig (plotly.graph_objs._figure.Figure): Generated plotly figure object.
        """

        assert (self.__latent_points.shape[-1] == 1 or self.__latent_points.shape[-1] == 2 or self.__latent_points.shape[-1] == 3) and \
            (self.__points_A.shape[-1] == 1 or self.__points_A.shape[-1] == 2 or self.__points_A.shape[-1] == 3) and \
            (self.__points_B.shape[-1] == 1 or self.__points_B.shape[-1] == 2 or self.__points_B.shape[-1] == 3), \
            "Only 1-, 2- or 3- dimensional visualizations are supported!"

        types = [None, None, None]

        types[0] = 'xy' if self.__latent_points.shape[-1] <= 2 else 'scene'
        types[1] = 'xy' if self.__points_A.shape[-1] <= 2 else 'scene'
        types[2] = 'xy' if self.__points_B.shape[-1] <= 2 else 'scene'

        fig = subplots.make_subplots(rows=1, cols=3, specs=[
                                     [{'type': item} for item in types]],
                                     subplot_titles=('latent', 'distribution A', 'distribution B'))

        def add_scatter(points: jnp.ndarray, levels: jnp.ndarray, row: int, col: int):
            data = {
                'mode': 'markers',
                'marker': {
                    'size': 2.5,
                    'color': levels
                },
            }
            
            if points.shape[-1] == 1:
                data.update({
                    'x': points[:, 1], 'y': jnp.zeros(points.shape[0])
                })
                
                fig.add_trace(go.Scatter(data), row=row, col=col)

                fig.update_xaxes(title_text='x', row=row, col=col)
                fig.update_yaxes(title_text='This axis is not used.', row=row, col=col)
                
            elif points.shape[-1] == 2:
                data.update({
                    'x': points[:, 0], 'y': points[:, 1],
                })
                
                fig.add_trace(go.Scatter(data), row=row, col=col)
                
                fig.update_xaxes(title_text='x', row=row, col=col)
                fig.update_yaxes(title_text='y', row=row, col=col)
                
            else:
                data.update(
                    {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
                )
                
                fig.add_trace(go.Scatter3d(data), row=row, col=col)
                
                fig.update_xaxes(title_text='x_0', row=row, col=col)
                fig.update_yaxes(title_text='x_1', row=row, col=col)

        levels = jnp.sqrt(
            jnp.sum(
                jnp.square(self.__latent_points),
                axis=-1))

        add_scatter(self.__latent_points, levels, 1, 1)
        add_scatter(self.__points_A, levels, 1, 2)
        add_scatter(self.__points_B, levels, 1, 3)

        return fig
