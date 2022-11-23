from abc import ABC, abstractmethod
from typing import Callable


class NetworkBlock(ABC):
    """Abstract base class for a regular neural network performing tasks such as regression or classification.

    Generators and discriminators
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compile_grad_fn(self, loss: Callable):
        """Returns the loss with respect to model parameters.

        This is doing symbolic computations.

        Returns:
            A function: (self.parameters_tree, batch) -> gradients with respect to the parameters
        """
        
        self.grad_fn = None
        self.gradients = None

        pass

    @abstractmethod
    def get_parameters(self):
        return self.parameters

    @abstractmethod
    def set_parameters(self, parameters):
        self.parameters = parameters
    
    @abstractmethod
    def update_gradients(self, batch):
        self.gradients = self.gradients

class Optimizer(ABC):
    
    def __init__(self, model: NetworkBlock):
        super().__init__()
    
    def step(batch):
        """Perform a gradient step

        Args:
            batch (_type_): _description_
        """
        
        gradients = model.grad_fn(model.get_parameters(), batch)
        ...


model = NetworkBlock()
gradient_fn = model.compile_grad_fn(loss)
grads = gradient_fn(model.parameters, batch)

model.apply_gradients(grads)

model.set_parameters(optimizer(model.get_parameters(), grads))
