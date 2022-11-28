from flax import linen as nn
import jax
from jax import random, numpy as jnp, tree_util
import optax
from typing import Tuple
from tqdm import tqdm
from math import sqrt

from .. import datasets
from .base import DifferentiableLearningSystem, ModelInstance


class VanillaGAN(DifferentiableLearningSystem):
    """

    """

    def __init__(self, generator: nn.Module, discriminator: nn.Module, latent_shape: Tuple[int], ambient_shape: Tuple[int]):
        """data_shape should NOT include a batch dimension
        """

        generator = ModelInstance(generator)
        discriminator = ModelInstance(discriminator)

        super().__init__((generator, discriminator))

        self.__latent_shape = latent_shape
        self.__ambient_shape = ambient_shape
        self.__generator = generator
        self.__discriminator = discriminator

    def initialize(self, loss_fn=optax.sigmoid_binary_cross_entropy):
        """1 denotes the TRUE samples!

        Args:
            loss_fn (Callable): The loss for the discriminator. This should operate on ONE y_pred, y_true pair
        """

        self.__generator.initialize(jnp.ones((1, *self.__latent_shape)))
        self.__discriminator.initialize(jnp.ones((1, *self.__ambient_shape)))

        # self.__generator.compile(loss_fn, need_vmap = True)
        self.__discriminator.compile(loss_fn, need_vmap=True)

    def train(self, dataset: datasets.base.TensorDataset,
              optimizer: optax.GradientTransformation,
              batch_size: int = 32,
              num_epochs: int = 10,
              key: random.KeyArray = random.PRNGKey(0),
              print_every: int = 10):

        self.__discriminator.attach_optimizer(optimizer)
        self.__generator.attach_optimizer(optimizer)

        forward_fn_gen = self.__generator.forward_fn
        forward_fn_dis = self.__discriminator.forward_fn

        @jax.jit
        def loss_fn_gen_combined(params_gen, state_gen, params_dis, state_dis, random_noise):
            """_summary_

            Args:
                params_gen (_type_): _description_
                state_gen (_type_): _description_
                params_dis (_type_): _description_
                state_dis (_type_): _description_
                random_noise (_type_): _description_

            Returns:
                fake samples, new state for generator
            """

            fake, new_state_gen = forward_fn_gen(
                params_gen, state_gen, random_noise)
            return -jnp.mean(jnp.log(nn.sigmoid(forward_fn_dis(params_dis, state_dis, fake)[0]))), new_state_gen

        gradient_fn_gen = jax.jit(
            jax.value_and_grad(loss_fn_gen_combined, has_aux=True))

        epochs = tqdm(range(num_epochs))

        key, new_key = random.split(key)
        dataloader = datasets.base.DataLoader(
            dataset, batch_size, new_key, auto_reshuffle=True)
        num_batches = dataloader.num_batches

        for epoch in epochs:
            batches = tqdm(dataloader)
            for i, x_batch in enumerate(batches):

                key, new_key1, new_key2 = random.split(key, 3)

                # random_noise = random.normal(
                #     new_key1, (batch_size, *self.__latent_shape))
                random_noise = random.uniform(
                    new_key1, (batch_size, *self.__latent_shape), minval=-1, maxval=1)

                fake = self.__generator(random_noise)
                labels_real = jnp.ones((batch_size,))
                labels_fake = jnp.zeros((batch_size,))

                # Update discriminator
                combined_batch, combined_labels = self.combine_datasets(
                    x_batch, fake, labels_real, labels_fake, new_key2)

                self.__discriminator.step(combined_batch, combined_labels)

                # Update generator
                (loss_gen, new_state_gen), grads = gradient_fn_gen(
                    self.__generator.parameters_,
                    self.__generator.state_,
                    self.__discriminator.parameters_,
                    self.__discriminator.state_,
                    random_noise
                )

                self.__generator.manual_step_with_optimizer(
                    grads, new_state_gen)

                if i % print_every == 0:
                    dis_loss = self.__discriminator.compute_loss(
                        combined_batch, combined_labels)
                    gen_grads = grads
                    dis_grads = self.__discriminator.eval_gradients(
                        combined_batch, combined_labels)

                    total_gen_elements = 0
                    total_gen_norm_squared = 0

                    def add_gen_elements(x: jnp.ndarray):
                        nonlocal total_gen_elements
                        total_gen_elements += jnp.size(x)

                    def add_gen_norm(x: jnp.ndarray):
                        nonlocal total_gen_norm_squared
                        total_gen_norm_squared += jnp.linalg.norm(x) ** 2

                    tree_util.tree_map(add_gen_elements, gen_grads)
                    tree_util.tree_map(add_gen_norm, gen_grads)

                    total_dis_elements = 0
                    total_dis_norm_squared = 0

                    def add_dis_elements(x: jnp.ndarray):
                        nonlocal total_dis_elements
                        total_dis_elements += jnp.size(x)

                    def add_dis_norm(x: jnp.ndarray):
                        nonlocal total_dis_norm_squared
                        total_dis_norm_squared += jnp.linalg.norm(x) ** 2

                    tree_util.tree_map(add_dis_elements, dis_grads)
                    tree_util.tree_map(add_dis_norm, dis_grads)

                    batches.set_description(
                        f'iteration {i}; gen_loss: {loss_gen: .2e}; dis_loss: {dis_loss: .2e}; gen_grads_magnitude: {sqrt(total_gen_norm_squared / total_gen_elements): .2e}; dis_grads_magnitude: {sqrt(total_dis_norm_squared / total_dis_elements): .2e}')
                    # batches.set_description(
                    #     f'iteration {i}; dis_loss: {dis_loss: .2e}; dis_grads_magnitude: {sqrt(total_dis_norm_squared / total_dis_elements): .2e}')

            # epochs.set_description(f'iteration {epoch}; gen_loss: {loss_gen}; dis_loss: {d_loss}')

    def combine_datasets(self, data_a, data_b, label_a: jnp.array, label_b: jnp.array, key: random.KeyArray = None):

        combined_examples = jnp.concatenate([data_a, data_b], axis=0)
        labels = jnp.concatenate([label_a, label_b], axis=0)

        # permutation = random.permutation(
        #     key, jnp.array(range(len(combined_examples))))

        # combined_examples = jnp.array(
        #     [combined_examples[index] for index in permutation])
        # labels = jnp.array([labels[index] for index in permutation])

        # return random.permutation(key, combined_examples, axis=0), random.permutation(key, labels, axis=0)
        return combined_examples, labels

    def create_distribution(self):
        """When sampling from the returned distribution, ALWAYS use standard normal

        Returns:
            _type_: _description_
        """

        return GANDistribution(self.__generator)

    def __str__(self):
        generator_str = str(self.__generator)
        discriminator_str = str(self.__discriminator)
        gen_lines = generator_str.split('\n')
        dis_lines = discriminator_str.split('\n')

        ret_lines = ['generator:']
        ret_lines += ['\t' + line for line in gen_lines]
        ret_lines.append('discriminator:')
        ret_lines += ['\t' + line for line in dis_lines]

        return '\n'.join(ret_lines)


class GANDistribution(datasets.base.Distribution):

    def __init__(self, generator: ModelInstance):
        super().__init__()

        self.__generator = generator

    def draw_samples(self, samples: jnp.array):
        """_summary_

        Args:
            samples (jnp.array): The random noise to use.

        Returns:
            _type_: _description_
        """
        return self.__generator(samples)
