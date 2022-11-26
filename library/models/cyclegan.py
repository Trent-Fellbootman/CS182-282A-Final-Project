from .base import DifferentiableLearningSystem, ModelInstance
from flax import linen as nn
import jax
from jax import random, numpy as jnp, tree_util
import optax
from optax._src.base import PyTree
from .. import datasets
from tqdm import tqdm
from typing import Tuple


class CycleGAN(DifferentiableLearningSystem):

    def __init__(self, generator: nn.Module, discriminator: nn.Module, data_shape: Tuple):
        generator_AB = ModelInstance(generator)
        generator_BA = ModelInstance(generator)

        discriminator_A =  ModelInstance(discriminator)
        discriminator_B =  ModelInstance(discriminator)

        super().__init__((generator_AB, generator_BA, discriminator_A, discriminator_B))

        self.__data_shape = data_shape

        self.__generator_AB = generator_AB
        self.__generator_BA = generator_BA
        self.__discriminator_A = discriminator_A
        self.__discriminator_B = discriminator_B

    def initialize(self):
        self.__generator_AB.initialize(jnp.ones(self.__data_shape))
        self.__generator_BA.initialize(jnp.ones(self.__data_shape))
        self.__discriminator_A.initialize(jnp.ones(self.__data_shape))
        self.__discriminator_B.initialize(jnp.ones(self.__data_shape))

        self.__discriminator_A.compile(optax.sigmoid_binary_cross_entropy, need_vmap=True)
        self.__discriminator_B.compile(optax.sigmoid_binary_cross_entropy, need_vmap=True)

    def train(self, dataset_A: datasets.base.TensorDataset,
              dataset_B: datasets.base.TensorDataset,
              num_epochs: int = 10,
              learning_rate: float = 1e-3,
              key: random.KeyArray = random.PRNGKey(0)):

        self.__discriminator_A.attach_optimizer(
            optax.sgd(learning_rate=learning_rate))

        self.__discriminator_B.attach_optimizer(
            optax.sgd(learning_rate=learning_rate))
        
        self.__generator_AB.attach_optimizer(
            optax.sgd(learning_rate=learning_rate)
        )

        self.__generator_BA.attach_optimizer(
            optax.sgd(learning_rate=learning_rate)
        )

        forward_fn_gen_AB = self.__generator_AB.forward_fn
        forward_fn_gen_BA = self.__generator_BA.forward_fn
        forward_fn_dis_A = self.__discriminator_A.forward_fn
        forward_fn_dis_B = self.__discriminator_B.forward_fn

        real_A = dataset_A
        real_B = dataset_B
        num_examples = len(dataset_A)

        @jax.jit
        # return the discriminator scores and the new state of the generator
        def forward_fn_AB(params_gen, state_gen, params_dis, state_dis, data):
            fake, new_state_gen = forward_fn_gen_AB(params_gen, state_gen, data)
            return -jnp.mean(forward_fn_dis_B(params_dis, state_dis, fake)[0]), new_state_gen

        def forward_fn_BA(params_gen, state_gen, params_dis, state_dis, data):
            fake, new_state_gen = forward_fn_gen_BA(params_gen, state_gen, data)
            return -jnp.mean(forward_fn_dis_A(params_dis, state_dis, fake)[0]), new_state_gen

        gradient_fn_gen_AB = jax.jit(jax.value_and_grad(forward_fn_AB, has_aux=True))
        gradient_fn_gen_BA = jax.jit(jax.value_and_grad(forward_fn_BA, has_aux=True))
        
        iterations = tqdm(range(num_epochs))
        for i in iterations:

            key, new_key, new_key2 = random.split(key, 3)
            
            # Fake samples from generators
            fake_A = self.__generator_BA(real_B)
            fake_B = self.__generator_AB(real_A)
            labels_real = jnp.ones((num_examples,))
            labels_fake = jnp.zeros((num_examples,))

            ## Update discriminators
            dA_dataset, dA_labels = self.combine_datasets(real_A, fake_A, labels_real, labels_fake, new_key)
            self.__discriminator_A.step(dA_dataset, dA_labels)
            dA_loss = self.__discriminator_A.compute_loss(dA_dataset, dA_labels)
            
            dB_dataset, dB_labels = self.combine_datasets(real_B, fake_B, labels_real, labels_fake, new_key2)
            self.__discriminator_B.step(dB_dataset, dB_labels)
            dB_loss = self.__discriminator_B.compute_loss(dB_dataset, dB_labels)

            ## Update Generators
            # Calculate GAN loss: Adversarial Loss

            # Cycle loss: Cycle Consistency Loss
            (loss_genAB, new_state_genAB), gradsAB = gradient_fn_gen_AB(
                self.__generator_AB.parameters_,
                self.__generator_AB.state_,
                self.__discriminator_B.parameters_,
                self.__discriminator_B.state_,
                real_A)
            
            (loss_genBA, new_state_genBA), gradsBA = gradient_fn_gen_BA(
                self.__generator_BA.parameters_,
                self.__generator_BA.state_,
                self.__discriminator_A.parameters_,
                self.__discriminator_A.state_,
                real_B)
            
            self.__generator_AB.manual_step_with_optimizer(gradsAB, new_state_genAB)
            self.__generator_BA.manual_step_with_optimizer(gradsBA, new_state_genBA)

            
            iterations.set_description(f'iteration {i}; gen_loss: {loss_genAB+loss_genBA}; dis_loss: {dA_loss+dB_loss}')


    def combine_datasets(self, data_a, data_b, label_a: jnp.array, label_b: jnp.array, key: random.KeyArray = None):
       
        combined_examples = jnp.concatenate([data_a, data_b], axis=0)
        labels = jnp.concatenate([label_a, label_b], axis=0)

        permutation = random.permutation(key, jnp.array(range(len(combined_examples))))

        combined_examples = jnp.array([combined_examples[index] for index in permutation])
        labels = jnp.array([labels[index] for index in permutation])

        return combined_examples, labels


    
    def create_distribution(self):
        return GANDistribution(self.__generator_AB)

class GANDistribution(datasets.base.Distribution):
    
    def __init__(self, generator: ModelInstance):
        super().__init__()
        
        self.__generator = generator
    
    def draw_samples(self, samples: jnp.array):
        return self.__generator(samples)