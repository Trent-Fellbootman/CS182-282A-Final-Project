from flax import linen as nn
import jax
from jax import random, numpy as jnp, tree_util
import optax
from typing import Tuple
from tqdm import tqdm

from .. import datasets
from .base import DifferentiableLearningSystem, ModelInstance



class VanillaGAN(DifferentiableLearningSystem):
    """

    """
    def __init__(self, generator: nn.Module, discriminator: nn.Module, data_shape: Tuple):
        generator = ModelInstance(generator)
        discriminator = ModelInstance(discriminator)

        super().__init__((generator, discriminator))

        self.__data_shape = data_shape
        self.__generator = generator
        self.__discriminator = discriminator

    def initialize(self, loss_fn, ):

        self.__generator.initialize(jnp.ones(self.__data_shape))
        self.__discriminator.initialize(jnp.ones(self.__data_shape))

        self.__generator.compile(loss_fn, need_vmap = True)
        self.__discriminator.compile(loss_fn, need_vmap = True)

    def train(self, dataset: datasets.base.TensorDataset,
              optimizer,
              num_epochs: int = 10,
              learning_rate: float = 1e-3, 
              key: random.KeyArray = random.PRNGKey(0)):

        self.__discriminator.attach_optimizer(optimizer(learning_rate = learning_rate))
        self.__generator.attach_optimizer(optimizer(learning_rate = learning_rate))

        forward_fn_gen = self.__generator.forward_fn
        forward_fn_dis = self.__discriminator.forward_fn


        @jax.jit
        def forward_fn_combined(params_gen, state_gen, params_dis, state_dis, random_noise):
            fake, new_state_gen = forward_fn_gen(params_gen, state_gen, random_noise)
            return -jnp.mean(forward_fn_dis(params_dis, state_dis, fake)[0]), new_state_gen

        gradient_fn_gen = jax.jit(jax.value_and_grad(forward_fn_combined, has_aux = True))

        iterations = tqdm(range(num_epochs))
        for i in iterations:
            key, new_key1, new_key2= random.split(key, 3)

            fake = self.__generator(dataset)
            labels_real = jnp.ones((len(dataset), ))
            labels_fake = jnp.ones((len(dataset), ))

            random_noise = random.normal(new_key1, self.__data_shape)

            # Update discriminator
            combined_dataset, combined_labels = self.combine_datasets(dataset, fake, labels_real, labels_fake, new_key2)
            self.__discriminator.step(combined_dataset, combined_labels)
            d_loss = self.__discriminator.compute_loss(combined_dataset, combined_labels)

            # Update generator
            (loss_gen, new_state_gen), grads = gradient_fn_gen(
                self.__generator.parameters_, 
                self.__generator.state_,
                self.__discriminator.parameters_, 
                self.__discriminator.state_,
                random_noise
            )

            self.__generator.manual_step_with_optimizer(grads, new_state_gen)
            iterations.set_description(f'iteration {i}; gen_loss: {loss_gen}; dis_loss: {d_loss}')
    def combine_datasets(self, data_a, data_b, label_a: jnp.array, label_b: jnp.array, key: random.KeyArray = None):
       
        combined_examples = jnp.concatenate([data_a, data_b], axis=0)
        labels = jnp.concatenate([label_a, label_b], axis=0)

        permutation = random.permutation(key, jnp.array(range(len(combined_examples))))

        combined_examples = jnp.array([combined_examples[index] for index in permutation])
        labels = jnp.array([labels[index] for index in permutation])

        return combined_examples, labels
    
    def create_distribution(self):
        return GANDistribution(self.__generator)

class GANDistribution(datasets.base.Distribution):
    
    def __init__(self, generator: ModelInstance):
        super().__init__()
        
        self.__generator = generator
    
    def draw_samples(self, samples: jnp.array):
        return self.__generator(samples)