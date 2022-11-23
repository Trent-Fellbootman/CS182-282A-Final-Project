from flax import linen as nn
from typing import *


class MLP(nn.Module):
    '''
    Standard Multi-Layer Perceptron architecture to use for generators and discriminators.
    '''
    hidden_sizes: Sequence[int]
    out_size: int
    apply_batchnorm: bool
    apply_layer_activation: bool
    output_activation: nn.Module

    def setup(self):
        self.layers = [nn.Dense(features=h) for h in self.hidden_sizes]
        self.output_layer = nn.Dense(features=self.out_size)
        if self.apply_layer_activation:
            self.layer_activation = nn.relu
        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm()


    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            #if self.apply_batchnorm:
            #    x = self.batchnorm(x, use_running_average=False)
            if self.apply_layer_activation:
                x = self.layer_activation(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

class Conv(nn.Module):
    '''
    Convolutional architecture to use for generators and discriminators.
    '''
    hidden_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    paddings: Sequence[int]
    apply_batchnorm: bool = False
    
    def setup(self):
        self.layers = [nn.Conv(features=self.hidden_sizes[i], 
        kernel_size=self.kernel_sizes[i],
        #padding=self.paddings[i],
        strides=self.strides[i]) for i in range(len(self.hidden_sizes))]
        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm()

    def __call__(self, images):
        x = images
        for layer in self.layers:
            x = layer(x)
            if self.apply_batchnorm:
                x = self.batchnorm(x)
        return x

def test_mlp():
    from jax import random
    mlp = MLP([16, 32], 2, True, True, nn.relu)
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (1,2))
    params = mlp.init(key2, x)
    print(x)
    y = mlp.apply(params, x)
    print(y)

def test_conv():
    from jax import random
    model = Conv([16], [(3,3)], [(2,2)], [(0,0)], False)
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (28, 28, 3))
    print(x.shape)
    params = model.init(key2, x)
    y = model.apply(params, x)
    print(y.shape)

if __name__ == "__main__":
    test_conv()