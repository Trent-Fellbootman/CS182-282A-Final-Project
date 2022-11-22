from typing import List, Tuple
from base import Dataset
import jax
from jax import numpy as jnp, random


class PointDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class Polynomial:
    """Callable polynomial class.
    """

    def __init__(self, specifiers: List[Tuple[float, List[int]]]):
        """Constructor.

        Args:
            `specifiers` (List[Tuple[float, List[int]]]): Specifies each term in the polynomial via `(coefficient, (variable 1 power, variable 2 power, ...))`
        """

        assert len(specifiers) > 0, "there must be at least one term!"

        self.__num_vars = len(specifiers[0][1])

        for _, exponents in specifiers:
            assert len(exponents) == self.__num_vars

        self.__specifiers = specifiers
    
    @property
    def term_count(self):
        """the number of terms of this polynomial.
        """
        
        return len(self.__specifiers)

    def __call__(self, X: jnp.ndarray):
        """batch-evaluate the polynomial.

        Args:
            X (jnp.ndarray): input values. Last axis refers to different entries. All other dimensions refer to batch dimensions.
        """

        assert len(X.shape) > 0 and X.shape[-1] == self.__num_vars, "Unmatched dimensionality of input data!"

        columns = [X[..., index] for index in range(self.__num_vars)]

        ret = jnp.zeros(X.shape[:-1])
        for coeff, exponents in self.__specifiers:
            term = jnp.ones_like(ret)
            for exponent, column in zip(exponents, columns):
                term = jnp.multiply(term, jnp.power(column, exponent))

            ret += coeff * term

        return ret

    @staticmethod
    def __enumerateAllTerms(dimension: int, max_order: int):
        """Enumerate all possible polynomial terms.
        
        You should NOT call this function directly.

        Args:
            dimension (int): the number of dimensions.
            max_order (int): the maximum order.
        """
        assert dimension > 0, "there must be at least 1 dimension!"
        assert max_order >= 0, "max_order must be nonnegative!"

        def _recurse(dimension: int, max_order: int):
            if dimension == 1:
                return [[max_order]]
            elif max_order == 0:
                return [[0] * dimension]
            else:
                ret = []
                for order in range(max_order + 1):
                    remains = _recurse(dimension - 1, max_order - order)
                    ret = [[order] + term for term in remains] + ret

                return ret

        allTerms = []
        for order in range(max_order + 1):
            allTerms += _recurse(dimension, order)

        return [tuple(term) for term in allTerms]

    @staticmethod
    def generateRandomPolynomial(dimension: int, max_order: int, coeff_range: float, key: random.KeyArray = random.PRNGKey(0)):
        """Generate a random polynomial.

        Args:
            `dimension` (int): The number of input dimensions.
            `max_order` (int): The maximum order of terms.
            `coeff_range` (float): The range of the coefficients. Coefficients will be distributed uniformly within `[-coeff_range, coeff_range)`.
            `key` (random.KeyArray, optional): The PRNG key to use. Defaults to random.PRNGKey(0).
        """

        allTerms = Polynomial.__enumerateAllTerms(dimension, max_order)

        return Polynomial(
            list(
                zip(
                    list(random.uniform(key, (len(allTerms),), minval=-coeff_range, maxval=coeff_range)),
                    allTerms)))
    
    def __str__(self):
        terms = []
        for coeff, exponents in self.__specifiers:
            sub_terms = []
            for i, exponent in enumerate(exponents):
                if exponent > 0:
                    sub_terms.append(f'x_{i}^{exponent}')

            if len(sub_terms) > 0:
                terms.append(f"{coeff:.2e} {' '.join(sub_terms)}")
            else:
                terms.append(f'{coeff:.2e}')
        
        return ' + '.join(terms)
