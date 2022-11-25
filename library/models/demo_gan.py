from .base import DifferentiableLearningSystem, ModelInstance
from .architectures import MLP

from .. import datasets

from flax import linen as nn
import jax
from jax import random, numpy as jnp, tree_util
import optax
from optax._src.base import PyTree
from tqdm import tqdm


class SimpleGANWithMLP(DifferentiableLearningSystem):

    def __init__(self, ambient_dim: int):
        generator = ModelInstance(MLP(hidden_sizes=[ambient_dim, ambient_dim, ambient_dim],
                                      out_size=ambient_dim,
                                      apply_batchnorm=False,
                                      apply_layer_activation=True,
                                      output_activation=nn.relu))

        discriminator = ModelInstance(MLP(hidden_sizes=[ambient_dim, ambient_dim, ambient_dim],
                                          out_size=1,
                                          apply_batchnorm=False,
                                          apply_layer_activation=True,
                                          output_activation=nn.relu))

        super().__init__((generator, discriminator))

        self.__ambient_dim = ambient_dim
        self.__latent_dim = None

        self.__generator = generator
        self.__discriminator = discriminator

    def initialize(self, latent_dim: int):
        self.__latent_dim = latent_dim

        self.__generator.initialize(jnp.ones((2, latent_dim)))
        self.__discriminator.initialize(jnp.ones((2, self.__ambient_dim)))

        self.__discriminator.compile(optax.sigmoid_binary_cross_entropy,
                                     need_vmap=True)

    def train(self, dataset: datasets.base.TensorDataset,
              num_epochs: int = 10,
              learning_rate: float = 1e-3,
              key: random.KeyArray = random.PRNGKey(0)):

        self.__discriminator.attach_optimizer(
            optax.sgd(learning_rate=learning_rate))
        
        self.__generator.attach_optimizer(
            optax.sgd(learning_rate=learning_rate)
        )

        true_examples = dataset.get_tensors()
        num_examples = len(dataset)

        forward_fn_gen = self.__generator.forward_fn
        forward_fn_dis = self.__discriminator.forward_fn

        @jax.jit
        # return the discriminator scores and the new state of the generator
        def forward_fn_combined(params_gen, state_gen, params_dis, state_dis, random_noise):
            fake, new_state_gen = forward_fn_gen(params_gen, state_gen, random_noise)
            return -jnp.mean(forward_fn_dis(params_dis, state_dis, fake)[0]), new_state_gen

        gradient_fn_gen = jax.jit(jax.value_and_grad(forward_fn_combined, has_aux=True))
        
        iterations = tqdm(range(num_epochs))
        for i in iterations:
            # generate fake samples
            key, new_key, new_key_2 = random.split(key, num=3)
            random_noise = random.normal(
                new_key, (num_examples, self.__ambient_dim))
            fake_samples = self.__generator(random_noise)

            combined_examples = jnp.concatenate(
                [true_examples, fake_samples], axis=0)
            labels = jnp.concatenate(
                [jnp.ones((num_examples,)), jnp.zeros((num_examples,))], axis=0)

            permutation = random.permutation(
                new_key_2, list(range(2 * num_examples)))

            combined_examples = jnp.array(
                [combined_examples[index] for index in permutation])
            labels = jnp.array([labels[index] for index in permutation])
            
            # update discriminator
            self.__discriminator.step(combined_examples, labels)
            dis_loss = self.__discriminator.compute_loss(combined_examples, labels)
            
            # update generator
            (loss_gen, new_state_gen), grads = gradient_fn_gen(self.__generator.parameters_,
                                                   self.__generator.state_,
                                                   self.__discriminator.parameters_,
                                                   self.__discriminator.state_,
                                                   random_noise)
            
            self.__generator.manual_step_with_optimizer(grads, new_state_gen)
            
            iterations.set_description(f'iteration {i}; gen_loss: {loss_gen}; dis_loss: {dis_loss}')
    
    def create_distribution(self):
        return GANDistribution(self.__generator, latent_dim=self.__latent_dim)

class GANDistribution(datasets.base.Distribution):
    
    def __init__(self, generator: ModelInstance, latent_dim: int):
        super().__init__()
        
        self.__generator = generator
        self.__latent_dim = latent_dim
    
    def draw_samples(self, n_samples: int, key: random.KeyArray = None):
        random_noise = random.normal(key, (n_samples, self.__latent_dim))
        return self.__generator(random_noise)