from jax import random, numpy as jnp
from flax import linen as nn
def check_model_forward(model: nn.Module, input_data: jnp.array):
    key = random.PRNGKey(0) # DO NOT CHANGE THIS
    test_input = random.normal(key, input_data.shape)
    params = model.init(key, test_input)
    return model.apply(params, test_input)