from flax import linen as nn, serialization
import jax
from jax import numpy as jnp, random, tree_util
from typing import Callable, Dict
import optax
from optax._src.base import GradientTransformation
from abc import ABC, abstractmethod
import copy


class ModelInstance:

    """This is a wrapper class that combines an nn.Module object
    (which is an "uninstantiated model template", since no parameters
    are included in it) and its parameters.

    This class is designed to be constructed ("instantiated") from an
    nn.Module object. It serves as a stateful wrapper, not an ABC
    (Abstract Base Class).

    This class is NOT designed to be composed. It represents a whole big
    model, not a component.
    """

    def __init__(self, template: nn.Module, batch_name: str = 'batch'):
        """Instantiates a concrete model.

        Args:
            template (nn.Module): An nn.Module object that defines the
            network structure. This includes hidden sizes, etc., but
            does not complete determines parameter shapes, since
            input shape is not yet determined.

            batch_name (str): The name of the batch dimension. This
            must be consistent with what is used in `template`, e.g.,
            the batch name argument passed in to the BatchNorm constructor.
        """

        self.__initialized = False
        self.__model_structure = template
        self.__parameters = None
        self.__state = None
        self.__batch_name = batch_name

        # self.__fast_apply is the JIT version of model.apply. The signature is (params, state, x_batch) -> (y_pred, new_state)
        self.__fast_apply = None

        # self.__loss_fn has signature (y_pred_batch, y_true_batch) -> float
        self.__loss_fn = None

        # self.__grad_fn has signature (params, state, x_batch, y_batch) -> gradients w.r.t. params, new_state.
        self.__grad_fn = None

        self.__optimizer = None
        self.__optimizer_state = None

        self.__run_configs = {}

    @property
    def batch_name(self):
        return self.__batch_name

    @property
    def variables(self):
        """Returns A COPY of the parameters and state variables.
        """

        if not self.__initialized:
            raise Exception(
                'This model is not initialized! Please call "initialize" first.')

        return tree_util.tree_map(lambda x: jnp.copy(x),
                                  {'params': self.__parameters, **self.__state})

    def update_configs(self, new_configs: Dict):
        """Updates the configurations that modifies the behavior
        of `model.apply`.

        This method does NOT reset the optimizer state.

        This method updates self.__configs (old configurations whose keys
        are not present in new_configs are retained), which will be passed in
        as additional named arguments when calling the apply method on
        on the nn.Module object. i.e., something like this will happen:

        ```
        model.apply({'params': params, **state}, x_batch, **self.configs, ...)
        ```

        Note that some nn.Module objects may have additional arguments
        that changes the behavior of model.apply. For example, BatchNorm has an
        additional argument called use_running_average, which determines whether
        or not the running averages of mean and variance will be updated.

        self.__configs should specify any additional arguments that you defined
        in the __call__ method of the nn.Module object. For example, if the
        nn.Module object you are wrapping is defined as:

        ```
        class MLP(nn.Module):
            hidden_size: int
            out_size: int

            @nn.compact
            def __call__(self, x, train=False):
                norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, axis_name='batch')

                x = nn.Dense(self.hidden_size)(x)
                x = norm()(x)
                x = nn.relu(x)
                x = nn.Dense(self.hidden_size)(x)
                x = norm()(x)
                x = nn.relu(x)
                x = nn.Dense(self.out_size)(x)

                return x
        ```

        Then, to set the model in training mode, you should call:

        ```
        model_instance.update_configs(self, {'train': True})
        ```

        Similarly, to set the model in evaluation mode, you should call:

        ```
        model_instance.update_configs(self, {'train': False})
        ```

        Args:
            `new_configs` (Dict): The new configurations to update.
        """

        self.__run_configs.update(new_configs)

        # recompile the model, since the behavior of `apply` may have been modified.
        if self.__initialized:
            self.compile(self.__loss_fn, need_vmap=False)

    def reset_configs(self, configs: Dict):
        """Similar to update_configs, but this method completely
        discards all old configurations.

        You may want to call this method to remove accidentally
        added key, value pairs.

        Args:
            configs (Dict)
        """

        self.__run_configs = configs

    @property
    def run_configs(self):
        """Returns A COPY of the configurations.
        """

        return copy.deepcopy(self.__run_configs)

    def intitialize(self, x: jnp.ndarray, key: random.KeyArray = random.PRNGKey(0)):
        """Initializes the model, inferencing the shapes of all parameters / variables
        and initializing their values.

        This function also compiles the forward pass.

        Args:
            x (jnp.ndarray): The input to use for shape interence.
            key: The PRNG key to use.
        """

        variables = self.__model_structure.init(key, x)
        self.__state, self.__parameters = variables.pop('params')

        self.__compile_forward()

        self.__initialized = True

    def __compile_forward(self):
        # vectorize and compile the forward pass
        def apply(params, state, x_batch):
            y_pred, new_state = self.__model_structure.apply({'params': params, **state},
                                                             x_batch, **self.__run_configs,
                                                             mutable=list(state.keys()))
            return y_pred, new_state

        self.__fast_apply = jax.jit(
            jax.vmap(apply, in_axes=(None, None, 0), out_axes=(
                0, None), axis_name=self.__batch_name)
        )

    def compile(self, loss_fn: Callable, need_vmap: bool = False, reduce_method: Callable = jnp.mean):
        """Symbolically vectorized and compiles the gradient computation graph
        with the loss function, as well as the forward pass.

        If the loss function changes, this method should be re-called.

        This function does nothing to the optimizer / optimizer state.

        Args:

            loss_fn (Callable): The loss function to use. The signature
            of this function should be
            (y_pred: jnp.ndarray, y_true: jnp.ndarray) -> loss: float.,

            need_vmap (bool): Whether loss_fn is defined for a batch
            or for one sample. Should be True if it is defined for one sample.,

            reduce_method (Callable): The method to use to reduce a batch of losses
            into a single float number, if automatic vmap were to happen. This
            argument is DISREGARDED if need_vmap == False.
        """

        if not self.__initialized:
            raise Exception(
                'This model is not initialized! Please call "initialize" first.')

        # vectorize and compile the forward pass
        self.__compile_forward()

        # compile the gradient function
        if need_vmap:
            vectorized_loss = jax.vmap(
                loss_fn, in_axes=0, out_axes=0, axis_name=self.__batch_name)

            def reduced_vectorized_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray):
                return reduce_method(vectorized_loss(y_pred, y_true))

        else:
            reduced_vectorized_loss = loss_fn

        self.__loss_fn = reduced_vectorized_loss

        def composed_loss(params, state, x_batch, y_batch):
            # y_pred, new_state = self.__model_structure.apply({'params': params, **state},
            #                                                  x_batch, **self.__run_configs,
            #                                                  mutable=list(state.keys()))
            y_pred, new_state = self.__fast_apply(params, state, x_batch)
            return reduced_vectorized_loss(y_pred, y_batch), new_state

        self.__grad_fn = jax.jit(
            jax.grad(composed_loss, argnums=0, has_aux=True))

    def __call__(self, x_batch: jnp.ndarray):
        """Applies the model instance to transform the inputs.

        This method performs only the forward pass, and does NOT update the
        parameters, state variables, or the optimizer state.

        Call this method ONLY IF you JUST want to evaluate the model
        (with current configuration, which may not be the test-time configuration)
        on a set of inputs and NOTHING ELSE.

        Args:
            x_batch (jnp.ndarray): Inputs.

        Returns: y_pred (jnp.ndarray): Transformed inputs.
        """

        if not self.__initialized:
            raise Exception(
                'This model is not initialized! Please call "initialize" first.')

        y_pred, new_state = self.__fast_apply(
            self.__parameters, self.__state, x_batch)

        return y_pred

    def eval_gradients(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        """Evaluates the gradients w.r.t. a batch.

        This function does NOT update the parameters or the state variables.

        Call this function ONLY if you just want to evaluate the gradients
        (e.g., when you want to see how the gradient behave at different
        values of `x` for debug purposes.)

        Args:
            `x_batch` (jnp.ndarray)

            `y_batch` (jnp.ndarray)

        Returns:
            gradients(pytree): Gradients w.r.t. to the parameters.
        """

        if self.__grad_fn is None:
            raise Exception(
                'The gradient function is not compiled! Please call "compile" first.')

        gradients, new_state = self.__grad_fn(
            self.__parameters, self.__state, x_batch, y_batch)

        return gradients

    def attach_optimizer(self, optimizer: GradientTransformation):
        """Attach and initialize an optimizer.

        If there is already an optimizer, the old optimizer is DISCARDED.

        Args:
            optimizer (GradientTransformation): The optimizer to attach.
        """

        if not self.__initialized:
            raise Exception(
                'This model is not initialized! Please call "initialize" first.')

        self.__optimizer = optimizer
        self.__optimizer_state = optimizer.init(self.__parameters)

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        """Takes an optimizer step.

        This method DOES update the parameters, state variables and
        optimizer state.

        Args:
            x_batch (jnp.ndarray)
            y_batch (jnp.ndarray)
        """
        if self.__grad_fn is None:
            raise Exception(
                'The gradient function is not compiled! Please call "compile" first.')

        if self.__optimizer is None:
            raise Exception(
                'This model has no optimizer attached to it! Please call "attach_optimizer" first.')

        gradients, new_state = self.__grad_fn(
            self.__parameters, self.__state, x_batch, y_batch)

        # update state variables
        self.__state = new_state

        # calculate the updates and new optimizer state
        updates, new_optimizer_state = self.__optimizer.update(
            gradients, self.__optimizer_state, self.__parameters)

        # update the optimizer state
        self.__optimizer_state = new_optimizer_state

        # update parameters
        self.__parameters = optax.apply_updates(self.__parameters, updates)

    def compute_loss(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        """Compute the loss.

        This method does NOT update the parameters, state variables or the optimizer state.

        Call this method ONLY IF you JUST want to evaluate the loss.

        Args:
            x_batch (jnp.ndarray)
            y_batch (jnp.ndarray)
        """
        
        if self.__grad_fn is None:
            raise Exception(
                'The gradient function is not compiled! Please call "compile" first.')

        y_pred = self(x_batch)

        return self.__loss_fn(y_pred, y_batch)

    def save_states(self, filepath: str | None = None):
        """Saves all states, including parameters, variables and optimizer state,
        into a dictionary.

        Args:
            filepath (str | None, optional): The file path to save to.
            If None, do not save to disk. Defaults to None.
        """

        state_dict = {
            'variables': {'params': self.__parameters, 'state': self.__state},
            'optimizer_state': self.__optimizer_state
        }

        state_dict_bytes = serialization.to_bytes(state_dict)

        if filepath is not None:
            with open(filepath, 'xb') as f:
                f.write(state_dict_bytes)

        return state_dict

    def load_states(self, states: str | Dict):
        """Loads parameters, variables and optimizer state.
        
        This method does not keep you from loading states when you have not
        attached an optimizer yet. When you re-attach a new optimizer, the
        optimizer states are overwritten. This thus allows you to load
        the states, but use a new optimizer or new losses.
        
        If you have already compiled the model and attached an optimizer,
        this method would load all model states and optimizer states.
        This is useful, for example, when you have to shutdown your computer
        but would like to resume training later (in which case you can save
        the states and load them back later on). 

        Args:
            filepath (str | Dict): State dictionary. If str, load state dict from file.
        """
        if not self.__initialized:
            raise Exception('Model is not initialized!')

        state_dict = {
            'variables': {'params': self.__parameters, 'state': self.__state},
            'optimizer_state': self.__optimizer_state
        }

        if isinstance(states, Dict):
            # TODO: optimize.
            state_dict = serialization.from_bytes(state_dict, serialization.to_bytes(states))
            
        elif isinstance(states, str):
            with open(states, 'rb') as f:
                state_dict = serialization.from_bytes(state_dict, f.read())
        else:
            raise Exception(f"Invalid argument type: {type(states)}")
        
        self.__parameters = state_dict['variables']['params']
        self.__state = state_dict['variables']['state']
        self.__optimizer_state = state_dict['optimizer_state']
