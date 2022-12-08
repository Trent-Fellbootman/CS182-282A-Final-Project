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

        self.__generator_AB_loss_fn = None
        self.__generator_BA_loss_fn = None

    def compile_generator_grad_fn(self, cycle_loss_weight: float = 1.0, cycle_loss_mask: str = 'None'):
        """Compiles the generator gradient function.

        Args:
            cycle_loss_weight (float, optional): The weight of the cycle loss. Defaults to 1.0.

            cycle_loss_mask (str, optional): Determines which cycle loss to mask out.
            'ABA': remove A->B->A ONLY;
            'BAB': remove B->A->B ONLY.
            Defaults to 'None'.
        """

        forward_fn_gen_AB = self.__generator_AB.forward_fn
        forward_fn_gen_BA = self.__generator_BA.forward_fn
        forward_fn_dis_A = self.__discriminator_A.forward_fn
        forward_fn_dis_B = self.__discriminator_B.forward_fn

        gan_weight, cycle_weight = 1.0 / \
            (1 + cycle_loss_weight), cycle_loss_weight / (1 + cycle_loss_weight)

        ##############################################
        # TODO: Define GAN and cycle-consistency     #
        # loss for each generator.                   #
        # HINT:                                      #
        # 1) Use forward_fn_dis to construct the     #
        # GAN loss. Remember that the function       #
        # outputs logits.                            #
        # 2) In the cycle loss compare the true      #
        # samples to the reconstructed samples.      #
        # 3) The generator loss should be the        #
        # average negative log-probability that      #
        # the samples generated are classified as    #
        # real by the discriminator                  #
        # 4) The cycle loss should be the average of #
        # the absolute difference between the        #
        # original and the reconstructed samples     #
        # (note that you need to take the average    #
        # over not only the samples, but also the    #
        # different entries in each sample)          #
        ##############################################

        @jax.jit
        def generator_AB_loss_fn(gen_AB_params, gen_AB_state, gen_BA_params, gen_BA_state, dis_B_params, dis_B_state, batch_A):
            fake_B, new_state_gen_AB = forward_fn_gen_AB(
                gen_AB_params, gen_AB_state, batch_A)
            recon_A, new_state_gen_BA = forward_fn_gen_BA(
                gen_BA_params, gen_BA_state, fake_B)

            gan_loss = -jnp.mean(
                jnp.log(
                    nn.sigmoid(
                        forward_fn_dis_B(
                            dis_B_params, dis_B_state, fake_B)[0])))  # TODO: Compute GAN loss

            cycle_loss = jnp.mean(
                jnp.abs(recon_A - batch_A)
            )  # TODO: Compute cycle loss

            cycle_weight_ABA = 0 if cycle_loss_mask == 'ABA' else cycle_weight

            return gan_weight * gan_loss + cycle_weight_ABA * cycle_loss, (fake_B, recon_A, gan_loss, cycle_loss, new_state_gen_AB)

        @jax.jit
        def generator_BA_loss_fn(gen_BA_params, gen_BA_state, gen_AB_params, gen_AB_state, dis_A_params, dis_A_state, batch_B):
            fake_A, new_state_gen_BA = forward_fn_gen_BA(
                gen_BA_params, gen_BA_state, batch_B)
            recon_B, new_state_gen_AB = forward_fn_gen_AB(
                gen_AB_params, gen_AB_state, fake_A)

            gan_loss = -jnp.mean(
                jnp.log(
                    nn.sigmoid(
                        forward_fn_dis_A(
                            dis_A_params, dis_A_state, fake_A)[0])))  # TODO: Compute GAN loss

            cycle_loss = jnp.mean(
                jnp.abs(recon_B - batch_B)
            )  # TODO: Compute cycle loss

            cycle_weight_BAB = 0 if cycle_loss_mask == 'BAB' else cycle_weight

            return gan_weight * gan_loss + cycle_weight_BAB * cycle_loss, (fake_A, recon_B, gan_loss, cycle_loss, new_state_gen_BA)

        ##############################################
        #               END OF YOUR CODE             #
        ##############################################

        self.__generator_AB_loss_fn = generator_AB_loss_fn
        self.__generator_BA_loss_fn = generator_BA_loss_fn

        @jax.jit
        def generator_loss_fn(gen_AB_params, gen_AB_state, gen_BA_params, gen_BA_state, dis_A_params, dis_A_state, dis_B_params, dis_B_state, batch_A, batch_B):
            gen_AB_loss, (fake_B, recon_A, gan_loss_AB, cycle_loss_AB, new_state_gen_AB) = self.__generator_AB_loss_fn(
                gen_AB_params, gen_AB_state, gen_BA_params, gen_BA_state, dis_B_params, dis_B_state, batch_A)
            gen_BA_loss, (fake_A, recon_B, gan_loss_BA, cycle_loss_BA, new_state_gen_BA) = self.__generator_BA_loss_fn(
                gen_BA_params, gen_BA_state, gen_AB_params, gen_AB_state, dis_A_params, dis_A_state, batch_B)
            return gen_AB_loss + gen_BA_loss, (fake_A, fake_B, recon_A, recon_B, gan_loss_AB, gan_loss_BA, cycle_loss_AB, cycle_loss_BA, new_state_gen_AB, new_state_gen_BA)

        grad_fn_generator = jax.jit(jax.value_and_grad(
            generator_loss_fn, argnums=(0, 2), has_aux=True))

        self.__grad_fn_generator = grad_fn_generator
    
    def eval_generator_grads(self, batch_A: jnp.ndarray, batch_B: jnp.ndarray):
        """Evaluates the gradients on the generator parameters. This method is const and does not change the state of this object.

        Args:
            batch_A (jnp.ndarray): _description_
            batch_B (jnp.ndarray): _description_
        
        Returns:
            (grads_generator_AB, grads_generator_BA)
        """
        
        (generator_loss_total,
                    (fake_A, fake_B, recon_A, recon_B, gan_loss_gen_AB, gan_loss_gen_BA, cycle_loss_gen_AB, cycle_loss_gen_BA, new_state_gen_AB, new_state_gen_BA)), \
                    (grads_gen_AB, grads_gen_BA) = \
                    self.__grad_fn_generator(self.__generator_AB.parameters_,
                                             self.__generator_AB.state_,
                                             self.__generator_BA.parameters_,
                                             self.__generator_BA.state_,
                                             self.__discriminator_A.parameters_,
                                             self.__discriminator_A.state_,
                                             self.__discriminator_B.parameters_,
                                             self.__discriminator_B.state_,
                                             batch_A, batch_B)
        
        return (grads_gen_AB, grads_gen_BA)

    def train(self, dataset_A: datasets.base.TensorDataset,
              dataset_B: datasets.base.TensorDataset,
              optimizer: optax.GradientTransformation = optax.adam(
                  learning_rate=1e-3),
              batch_size: int = 32,
              num_epochs: int = 10,
              key: random.KeyArray = random.PRNGKey(0),
              print_every: int = 10,
              cycle_loss_weight: float = 1.0, cycle_loss_mask: str = 'None'):
        """Cycle loss mask determines which cycle loss to mask out. 'ABA': remove A->B->A ONLY; 'BAB': remove B->A->B ONLY.
        
        This method automatically compiles the gradient function; you do NOT need to manually call `compile_generator_grad_fn`.

        Args:
            dataset_A (datasets.base.TensorDataset): _description_
            dataset_B (datasets.base.TensorDataset): _description_
            optimizer (optax.GradientTransformation, optional): _description_. Defaults to optax.adam( learning_rate=1e-3).
            batch_size (int, optional): _description_. Defaults to 32.
            num_epochs (int, optional): _description_. Defaults to 10.
            key (random.KeyArray, optional): _description_. Defaults to random.PRNGKey(0).
            print_every (int, optional): _description_. Defaults to 10.
            cycle_loss_weight (float, optional): _description_. Defaults to 1.0.
            cycle_loss_mask (str, optional): _description_. Defaults to 'None'.

        Returns:
            Loss history.
        """
        
        self.compile_generator_grad_fn(cycle_loss_weight=cycle_loss_weight, cycle_loss_mask=cycle_loss_mask)

        self.__discriminator_A.attach_optimizer(optimizer)
        self.__discriminator_B.attach_optimizer(optimizer)
        self.__generator_AB.attach_optimizer(optimizer)
        self.__generator_BA.attach_optimizer(optimizer)

        epochs = tqdm(range(num_epochs))
        generator_loss_total = []
        discriminator_loss_total = []
        key, new_key = random.split(key)
        dataset = datasets.base.TensorDataset(
            (dataset_A.get_tensors(), dataset_B.get_tensors()))
        dataloader = datasets.base.DataLoader(
            dataset, batch_size, new_key, auto_reshuffle=True)

        gen_AB_cycle_losses = []
        gen_AB_gan_losses = []
        gen_BA_cycle_losses = []
        gen_BA_gan_losses = []
        dis_A_losses = []
        dis_B_losses = []

        for epoch in epochs:
            batches = tqdm(dataloader)
            for i, (batch_A, batch_B) in enumerate(batches):

                key, new_key_1, new_key2 = random.split(key, 3)

                real_A = batch_A
                real_B = batch_B

                (generator_loss_total,
                    (fake_A, fake_B, recon_A, recon_B, gan_loss_gen_AB, gan_loss_gen_BA, cycle_loss_gen_AB, cycle_loss_gen_BA, new_state_gen_AB, new_state_gen_BA)), \
                    (grads_gen_AB, grads_gen_BA) = \
                    self.__grad_fn_generator(self.__generator_AB.parameters_,
                                             self.__generator_AB.state_,
                                             self.__generator_BA.parameters_,
                                             self.__generator_BA.state_,
                                             self.__discriminator_A.parameters_,
                                             self.__discriminator_A.state_,
                                             self.__discriminator_B.parameters_,
                                             self.__discriminator_B.state_,
                                             batch_A, batch_B)

                # update generators
                self.__generator_AB.manual_step_with_optimizer(
                    grads_gen_AB, new_state_gen_AB)
                self.__generator_BA.manual_step_with_optimizer(
                    grads_gen_BA, new_state_gen_BA)

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

                # logs
                gen_AB_cycle_losses.append(cycle_loss_gen_AB.item())
                gen_AB_gan_losses.append(gan_loss_gen_AB.item())
                gen_BA_cycle_losses.append(cycle_loss_gen_BA.item())
                gen_BA_gan_losses.append(gan_loss_gen_BA.item())
                dis_A_losses.append(self.__discriminator_A.compute_loss(
                    dA_batch, dA_labels).item())
                dis_B_losses.append(self.__discriminator_B.compute_loss(
                    dB_batch, dB_labels).item())

                if i % print_every == 0:
                    dA_loss = self.__discriminator_A.compute_loss(
                        dA_batch, dA_labels)
                    dB_loss = self.__discriminator_B.compute_loss(
                        dB_batch, dB_labels)

                    # dis_grads = self.__discriminator_A.eval_gradients(dA_batch, dA_labels) + self.__discriminator_B.eval_gradients(dB_batch, dB_labels)
                    dis_grads_A = self.__discriminator_A.eval_gradients(
                        dA_batch, dA_labels)
                    dis_grads_B = self.__discriminator_B.eval_gradients(
                        dB_batch, dB_labels)

                    loss_dis_total = dA_loss + dB_loss
                    #loss_gen = loss_genAB + loss_genBA
                    #gen_grads = gradsAB + gradsBA

                    total_gen_AB_elements = 0
                    total_gen_AB_norm_squared = 0

                    def add_gen_AB_elements(x: jnp.ndarray):
                        nonlocal total_gen_AB_elements
                        total_gen_AB_elements += jnp.size(x)

                    def add_gen_AB_norm(x: jnp.ndarray):
                        nonlocal total_gen_AB_norm_squared
                        total_gen_AB_norm_squared += jnp.linalg.norm(x) ** 2

                    total_gen_BA_elements = 0
                    total_gen_BA_norm_squared = 0

                    def add_gen_BA_elements(x: jnp.ndarray):
                        nonlocal total_gen_BA_elements
                        total_gen_BA_elements += jnp.size(x)

                    def add_gen_BA_norm(x: jnp.ndarray):
                        nonlocal total_gen_BA_norm_squared
                        total_gen_BA_norm_squared += jnp.linalg.norm(x) ** 2

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

                    tree_util.tree_map(
                        add_gen_AB_elements, grads_gen_AB)
                    tree_util.tree_map(
                        add_gen_AB_norm, grads_gen_AB)

                    tree_util.tree_map(
                        add_gen_BA_elements, grads_gen_BA)
                    tree_util.tree_map(
                        add_gen_BA_norm, grads_gen_BA)

                    tree_util.tree_map(add_dis_A_elements, dis_grads_A)
                    tree_util.tree_map(add_dis_A_norm, dis_grads_A)

                    tree_util.tree_map(add_dis_B_elements, dis_grads_B)
                    tree_util.tree_map(add_dis_B_norm, dis_grads_B)

                    batches.set_description(
                        f'Epoch {epoch}; Generator AB GAN loss: {gan_loss_gen_AB: .4f}; Generator BA GAN loss: {gan_loss_gen_BA: .4f}; B->A->B cycle loss: {cycle_loss_gen_BA: .4f}; A->B->A cycle loss: {cycle_loss_gen_AB: .4f}; Discriminator A loss: {dA_loss: .4f}; Discriminator B loss: {dB_loss: .4f}')

                    #grad_magnitude_gen_AB = sqrt(total_gen_AB_norm_squared / total_gen_AB_elements)
                    #grad_magnitude_gen_BA = sqrt(total_gen_BA_norm_squared / total_gen_BA_elements)
                    #grad_magnitude_dis_A = sqrt(total_dis_A_norm_squared / total_dis_A_elements)
                    #grad_magnitude_dis_B = sqrt(total_dis_B_norm_squared / total_dis_B_elements)
                    # batches.set_description(
                    #    f'epoch {epoch}; g_AB_l_g: {gan_loss_gen_AB: .2e} g_AB_l_c: {cycle_loss_gen_AB: .2e}; g_BA_l_g: {gan_loss_gen_BA: .2e} g_BA_l_c: {cycle_loss_gen_BA: .2e}; d_A_l: {dA_loss: .2e}; d_B_l: {dB_loss: .2e}; g_AB_gm: {grad_magnitude_gen_AB: .2e}; g_BA_gm: {grad_magnitude_gen_BA: .2e}; dis_A_gm: {grad_magnitude_dis_A: .2e}; dis_B_gm: {grad_magnitude_dis_B: .2e}')

        return {
            'gen_AB_cycle_losses': gen_AB_cycle_losses,
            'gen_AB_gan_losses': gen_AB_gan_losses,
            'gen_BA_cycle_losses': gen_BA_cycle_losses,
            'gen_BA_gan_losses': gen_BA_gan_losses,
            'dis_A_losses': dis_A_losses,
            'dis_B_losses': dis_B_losses
        }

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

    @property
    def generator(self):
        """Returns the underlying generator.
        """

        return self.__generator

    def draw_samples(self, samples: jnp.array):
        """_summary_

        Args:
            samples (jnp.array): The input data to use

        Returns:
            _type_: _description_
        """
        return self.__generator(samples)
