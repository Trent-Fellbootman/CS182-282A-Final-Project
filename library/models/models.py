from flax import linen as nn
from architectures import *
import optax


class CycleGAN:

    def setup(self):
        self.G_XtoY = MLP()

        loss = # TODO
        pass

    def forward(self, x):
        pass

    def backward(self):
        pass


class GAN:
    generator_hidden_sizes: Sequence[int]
    generator_output_dim: int
    discriminator_hidden_sizes: Sequence[int]
    discriminator_output_dim: int

    def setup(self):
        self.G = MLP(self.generator_hidden_sizes, self.generator_output_dim, True, nn.tanh)
        self.D = MLP(self.discriminator_hidden_sizes, self.discriminator_output_dim, False, nn.sigmoid)

    def forward(self, x):
        pass
    
    def backward(Self, x):
        pass
