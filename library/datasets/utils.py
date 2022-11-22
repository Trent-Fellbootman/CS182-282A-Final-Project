from jax import numpy as jnp, random
from typing import List, Tuple
from base import Dataset

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
    
    @property
    def input_dimension(self):
        return self.__num_vars

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

class PolynomialPointSampler:
    """Multioutput polynomial class.
    """
    
    def __init__(self, *polynomials: Polynomial):
        """Constructor.
        
        Args:
            *polynomials (Polynomial): The polynomials at each output dimension.
        """
        
        assert len(polynomials) > 0, "There must be at least one polynomial!"
        
        self.__input_dimension = polynomials[0].input_dimension
        self.__output_dimension = len(polynomials)
        
        for polynomial in polynomials:
            assert self.__input_dimension == polynomial.input_dimension
        
        self.__polynomials = tuple(polynomials)
    
    def __call__(self, X: jnp.ndarray):
        """batch-evaluate the multi-output polynomial.

        Args:
            X (jnp.ndarray): input values. Last axis refers to different entries. All other dimensions refer to batch dimensions.
        """
        
        return jnp.stack([polynomial(X) for polynomial in self.__polynomials], axis=-1)
    
    @property
    def output_dimension(self):
        return self.__output_dimension
    
    @property
    def input_dimension(self):
        return self.__input_dimension
    
    def generateRandomSampler(input_dimension: int, output_dimension: int,
                                 max_order: int, coeff_range: float, key: random.KeyArray = random.PRNGKey(0)):
        """Generate a random multi-output polynomial.

        Args:
            `input_dimension` (int): The number of input dimensions.
            `output_dimension` (int): The number of output dimensions.
            `max_order` (int): The maximum order of polynomial terms.
            `coeff_range` (float): The range of the coefficients. Coefficients will be distributed uniformly within `[-coeff_range, coeff_range)`.
            `key` (random.KeyArray, optional): The PRNG key to use. Defaults to random.PRNGKey(0).
        """
        
        keys = tuple(random.split(key, num=output_dimension))
        
        polynomials = [Polynomial.generateRandomPolynomial(input_dimension, max_order, coeff_range, rng_key) for rng_key in keys]
        
        return PolynomialPointSampler(*polynomials)
    
    def __str__(self):
        return '\n'.join([str(poly) for poly in self.__polynomials])

class TensorDataset(Dataset):
    """Tensor dataset.
    """
    
    def __init__(self, *tensors: List[jnp.ndarray]):
        """Constructor.
        
        Args:
            `*tensors`: A list of tensors. All tensors must have the same size in the first dimension, which is treated as the batch dimension.
        """
        assert len(tensors) > 0, "There must be at least one tensor!"
        
        self.__num_samples = tensors[0].shape[0]
        
        for tensor in tensors:
            assert tensor.shape[0] == self.__num_samples
        
        super().__init__()
        
        self.__tensors = tuple(tensors)
    
    def __getitem__(self, index: int):
        return tuple(tensor[index] for tensor in self.__tensors)
    
    def __len__(self):
        return self.__num_samples