from .base import DifferentiableLearningSystem, ModelInstance
from flax import linen as nn
import jax
from jax import random, numpy as jnp, tree_util
import optax
from optax._src.base import PyTree
from .. import datasets
from tqdm import tqdm
from typing import Tuple, Dict, List, Any, Callable
from math import sqrt


class CycleGAN(DifferentiableLearningSystem):

    # def __init__(self, generator: nn.Module, discriminator: nn.Module, data_shape_A: Tuple, data_shape_B: Tuple):
    def __init__(self, modules: Dict, data_shape_A: Tuple[int], data_shape_B: Tuple[int]):
        """Constructor.

        Args:
            modules (Dict): A dictionary containing generators and discriminators. Sturcture:
            {
                'generator_AB': nn.Module,
                'generator_BA': nn.Module,
                'discriminator_A': nn.Module,
                'discriminator_B': nn.Module,
            }
        """

        generator_AB = modules['generator_AB']
        generator_BA = modules['generator_BA']
        discriminator_A = modules['discriminator_A']
        discriminator_B = modules['discriminator_B']

        generator_AB_instance = ModelInstance(generator_AB)
        generator_BA_instance = ModelInstance(generator_BA)

        discriminator_A_instance = ModelInstance(discriminator_A)
        discriminator_B_instance = ModelInstance(discriminator_B)

        super().__init__((generator_AB_instance, generator_BA_instance,
                          discriminator_A_instance, discriminator_B_instance))

        self.__data_shape_A = data_shape_A
        self.__data_shape_B = data_shape_B

        self.__generator_AB = generator_AB_instance
        self.__generator_BA = generator_BA_instance
        self.__discriminator_A = discriminator_A_instance
        self.__discriminator_B = discriminator_B_instance

    def initialize(self, loss_fn):
        """Loss function SHOULD need vmap

        Args:
            loss_fn (_type_): _description_
        """

        self.__generator_AB.initialize(jnp.ones((1, *self.__data_shape_A)))
        self.__generator_BA.initialize(jnp.ones((1, *self.__data_shape_B)))
        self.__discriminator_A.initialize(jnp.ones((1, *self.__data_shape_A)))
        self.__discriminator_B.initialize(jnp.ones((1, *self.__data_shape_B)))

        self.__discriminator_A.compile(loss_fn, need_vmap=True)
        self.__discriminator_B.compile(loss_fn, need_vmap=True)

    def train(self, dataset_A: datasets.base.TensorDataset,
              dataset_B: datasets.base.TensorDataset,
              optimizer: optax.GradientTransformation = optax.adam(
                  learning_rate=1e-3),
              batch_size: int = 32,
              num_epochs: int = 10,
              key: random.KeyArray = random.PRNGKey(0),
              print_every: int = 10):

        self.__discriminator_A.attach_optimizer(optimizer)
        self.__discriminator_B.attach_optimizer(optimizer)
        self.__generator_AB.attach_optimizer(optimizer)
        self.__generator_BA.attach_optimizer(optimizer)

        forward_fn_gen_AB = self.__generator_AB.forward_fn
        forward_fn_gen_BA = self.__generator_BA.forward_fn
        forward_fn_dis_A = self.__discriminator_A.forward_fn
        forward_fn_dis_B = self.__discriminator_B.forward_fn

        # @jax.jit
        # def generatorAB_loss_fn(params_gen, state_gen, params_dis, state_dis, data):
        #    fake, new_state_gen = forward_fn_gen_AB(params_gen, state_gen, data)
        #    recov = self.__generator_BA(fake)
        #    return -jnp.mean(jnp.log(nn.sigmoid(forward_fn_dis_B(params_dis, state_dis, fake)[0]))) + jnp.sum(jnp.abs(data-recov)), new_state_gen

        # @jax.jit
        # def generatorBA_loss_fn(params_gen, state_gen, params_dis, state_dis, data):
        #    fake, new_state_gen = forward_fn_gen_BA(params_gen, state_gen, data)
        #    recov = self.__generator_AB(fake)
        #    return -jnp.mean(jnp.log(nn.sigmoid(forward_fn_dis_A(params_dis, state_dis, fake)[0]))) + jnp.sum(jnp.abs(data-recov)), new_state_gen

        #gradient_fn_gen_AB = jax.jit(jax.value_and_grad(generatorAB_loss_fn, has_aux=True))
        #gradient_fn_gen_BA = jax.jit(jax.value_and_grad(generatorBA_loss_fn, has_aux=True))

        @jax.jit
        def generator_loss_fn(gen_params: Tuple[Any], gen_states: Tuple[Any], dis_params: Tuple[Any], dis_states: Tuple[Any], batch_A, batch_B):
            generator_AB_params, generator_BA_params = gen_params
            generator_AB_state, generator_BA_state = gen_states
            
            discriminator_A_params, discriminator_B_params = dis_params
            discriminator_A_state, discriminator_B_state = dis_states
            
            fake_B, fake_state_genAB = forward_fn_gen_AB(
                generator_AB_params, generator_AB_state, batch_A)
            fake_A, fake_state_genBA = forward_fn_gen_BA(
                generator_BA_params, generator_BA_state, batch_B)
            # TODO: Unsure whether BA and AB states should be reversed below
            recov_A, recov_state_genBA = forward_fn_gen_BA(
                generator_BA_params, fake_state_genBA, fake_B)
            recov_B, recov_state_genAB = forward_fn_gen_AB(
                generator_AB_params, fake_state_genAB, fake_A)

            gan_loss = -(
                jnp.mean(
                    jnp.log(
                        nn.sigmoid(
                            forward_fn_dis_A(
                                discriminator_A_params, discriminator_A_state, fake_A)[0])))
                + jnp.mean(
                    jnp.log(
                        nn.sigmoid(
                            forward_fn_dis_B(
                                discriminator_B_params, discriminator_B_state, fake_B)[0]))))

            # cycle_loss = \
            #     jnp.mean(
            #         jnp.sum(
            #             jnp.abs(batch_A - recov_A),
            #             axis=-1)) \
            #     + jnp.mean(
            #         jnp.sum(
            #             jnp.abs(batch_B - recov_B),
            #             axis=-1))

            cycle_loss = \
                jnp.mean(
                    jnp.abs(batch_A - recov_A)) \
                + jnp.mean(
                    jnp.abs(batch_B - recov_B))

            # return gan_loss - cycle_loss, (recov_state_genBA, recov_state_genAB)
            return (gan_loss + cycle_loss), (fake_A, fake_B, recov_state_genAB, recov_state_genBA)

        gradient_fn_gen = jax.jit(
            jax.value_and_grad(
                generator_loss_fn, has_aux=True))

        epochs = tqdm(range(num_epochs))
        generator_loss_total = []
        discriminator_loss_total = []
        key, new_key = random.split(key)
        dataset = datasets.base.TensorDataset((dataset_A.get_tensors(), dataset_B.get_tensors()))
        dataloader = datasets.base.DataLoader(
            dataset, batch_size, new_key, auto_reshuffle=True)

        for epoch in epochs:
            batches = tqdm(dataloader)
            for i, (batch_A, batch_B) in enumerate(batches):

                key, new_key_1, new_key2 = random.split(key, 3)

                real_A = batch_A 
                real_B = batch_B
                
                (gen_loss_total, (fake_A, fake_B, recov_state_genAB, recov_state_genBA)), \
                    (grads_gen_AB, grads_gen_BA) = \
                        gradient_fn_gen(
                            (self.__generator_AB.parameters_,
                             self.__generator_BA.parameters_),

                            (self.__generator_AB.state_,
                             self.__generator_BA.state_),

                            (self.__discriminator_A.parameters_,
                             self.__discriminator_B.parameters_),
                            
                            (self.__discriminator_A.state_,
                             self.__discriminator_B.state_),
                            
                            batch_A, batch_B
                        )
                
                # update generators
                self.__generator_AB.manual_step_with_optimizer(
                    grads_gen_AB, recov_state_genAB)
                self.__generator_BA.manual_step_with_optimizer(
                    grads_gen_BA, recov_state_genBA)

                # Fake samples from generators
                # fake_A = self.__generator_BA(real_B)
                # fake_B = self.__generator_AB(real_A)
                
                labels_real = jnp.ones((batch_size,))
                labels_fake = jnp.zeros((batch_size,))

                # Update discriminators
                dA_batch, dA_labels = self.combine_batches(
                    real_A, fake_A, labels_real, labels_fake, new_key_1)

                # update discriminator A
                self.__discriminator_A.step(dA_batch, dA_labels)

                dB_batch, dB_labels = self.combine_batches(
                    real_B, fake_B, labels_real, labels_fake, new_key2)
                
                # update discriminator B
                self.__discriminator_B.step(dB_batch, dB_labels)

                # Update Generators
                # (loss_genAB, new_state_genAB), gradsAB = gradient_fn_gen_AB(
                #    self.__generator_AB.parameters_,
                #    self.__generator_AB.state_,
                #    self.__discriminator_B.parameters_,
                #    self.__discriminator_B.state_,
                #    real_A)
                #
                # (loss_genBA, new_state_genBA), gradsBA = gradient_fn_gen_BA(
                #    self.__generator_BA.parameters_,
                #    self.__generator_BA.state_,
                #    self.__discriminator_A.parameters_,
                #    self.__discriminator_A.state_,
                #    real_B)
                # gen_params = (self.__generator_AB.parameters_, self.__generator_BA.parameters_,
                #               self.__discriminator_A.parameters_, self.__discriminator_B.parameters_)
                # gen_states = (self.__generator_AB.state_, self.__generator_BA.state_,
                #               self.__discriminator_A.state_, self.__discriminator_A.state_)
                # (loss_gen, (new_state_genBA, new_state_genAB)), grads = gradient_fn_gen(
                #     real_A, real_B, gen_params, gen_states)

                if i % print_every == 0:
                    dA_loss = self.__discriminator_A.compute_loss(
                        dA_batch, dA_labels)
                    dB_loss = self.__discriminator_B.compute_loss(
                        dB_batch, dB_labels)

                    # dis_grads = self.__discriminator_A.eval_gradients(dA_batch, dA_labels) + self.__discriminator_B.eval_gradients(dB_batch, dB_labels)
                    dis_grads_A = self.__discriminator_A.eval_gradients(dA_batch, dA_labels)
                    dis_grads_B = self.__discriminator_B.eval_gradients(dB_batch, dB_labels)

                    loss_dis_total = dA_loss + dB_loss
                    #loss_gen = loss_genAB + loss_genBA
                    #gen_grads = gradsAB + gradsBA

                    total_gen_elements = 0
                    total_gen_norm_squared = 0

                    def add_gen_elements(x: jnp.ndarray):
                        nonlocal total_gen_elements
                        total_gen_elements += jnp.size(x)

                    def add_gen_norm(x: jnp.ndarray):
                        nonlocal total_gen_norm_squared
                        total_gen_norm_squared += jnp.linalg.norm(x) ** 2

                    #tree_util.tree_map(add_gen_elements, gen_grads)
                    #tree_util.tree_map(add_gen_norm, gen_grads)

                    total_dis_A_elements = 0
                    total_dis_A_norm_squared = 0

                    def add_dis_A_elements(x: jnp.ndarray):
                        nonlocal total_dis_A_elements
                        total_dis_A_elements += jnp.size(x)

                    def add_dis_A_norm(x: jnp.ndarray):
                        nonlocal total_dis_A_norm_squared
                        total_dis_A_norm_squared += jnp.linalg.norm(x) ** 2
                        
                    total_dis_B_elements = 0
                    total_dis_B_norm_squared = 0

                    def add_dis_B_elements(x: jnp.ndarray):
                        nonlocal total_dis_B_elements
                        total_dis_B_elements += jnp.size(x)

                    def add_dis_B_norm(x: jnp.ndarray):
                        nonlocal total_dis_B_norm_squared
                        total_dis_B_norm_squared += jnp.linalg.norm(x) ** 2

                    tree_util.tree_map(add_gen_elements, (grads_gen_AB, grads_gen_BA))
                    tree_util.tree_map(add_gen_norm, (grads_gen_AB, grads_gen_BA))
                    
                    tree_util.tree_map(add_dis_A_elements, dis_grads_A)
                    tree_util.tree_map(add_dis_A_norm, dis_grads_A)
                    
                    tree_util.tree_map(add_dis_B_elements, dis_grads_B)
                    tree_util.tree_map(add_dis_B_norm, dis_grads_B)

                    batches.set_description(
                        f'iteration {i}; gen_loss: {gen_loss_total: .2e}; dis_A_loss: {dA_loss: .2e}; dis_B_loss: {dB_loss: .2e}')
                    generator_loss_total.append(gen_loss_total)
                    discriminator_loss_total.append(loss_dis_total)
        return generator_loss_total, discriminator_loss_total

    def combine_batches(self, data_a, data_b, label_a: jnp.array, label_b: jnp.array, key: random.KeyArray = None):
        combined_examples = jnp.concatenate([data_a, data_b], axis=0)
        labels = jnp.concatenate([label_a, label_b], axis=0)
        
        return random.permutation(key, combined_examples), random.permutation(key, labels)

    def create_distribution(self):
        """When sampling from the returned distribution, ALWAYS use standard normal

        Returns:
            _type_: _description_
        """

        return GANDistribution(self.__generator_AB), GANDistribution(self.__generator_BA)

    def __str__(self):
        generator_AB_str = str(self.__generator_AB)
        generator_BA_str = str(self.__generator_BA)
        discriminator_A_str = str(self.__discriminator_A)
        discriminator_B_str = str(self.__discriminator_B)

        gen_AB_lines = generator_AB_str.split('\n')
        gen_BA_lines = generator_BA_str.split('\n')
        dis_A_lines = discriminator_A_str.split('\n')
        dis_B_lines = discriminator_B_str.split('\n')

        ret_lines = ['generator_AB:']
        ret_lines += ['\t' + line for line in gen_AB_lines]
        ret_lines.append('generator_BA:')
        ret_lines += ['\t' + line for line in gen_BA_lines]
        ret_lines.append('discriminator_A:')
        ret_lines += ['\t' + line for line in dis_A_lines]
        ret_lines.append('discriminator_B:')
        ret_lines += ['\t' + line for line in dis_B_lines]

        return '\n'.join(ret_lines)


class GANDistribution(datasets.base.Distribution):

    def __init__(self, generator: ModelInstance):
        super().__init__()

        self.__generator = generator

    def draw_samples(self, samples: jnp.array):
        """_summary_

        Args:
            samples (jnp.array): The input data to use

        Returns:
            _type_: _description_
        """
        return self.__generator(samples)
