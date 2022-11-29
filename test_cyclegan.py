from library import datasets, models
from flax import linen as nn
from jax import random
import plotly.io as pio
import optax
pio.renderers.default = 'notebook_connected'
MODEL_LATENT_DIM = 4
TRUE_LATENT_DIM = 2
AMBIENT_DIM = 2

key = random.PRNGKey(5)
key_A, key_B = random.split(key)
A = datasets.utils.make_blobs(n_samples=1000, min_sigma=0, max_sigma=0.1, key=key_A)
B = datasets.utils.make_blobs(n_samples=1000, min_sigma=0, max_sigma=0.1, key=key_B)

discriminator = nn.Sequential([
    nn.Dense(8),
    nn.relu,
    nn.Dense(16),
    nn.relu,
    nn.Dense(16),
    nn.relu,
    nn.Dense(8),
    nn.relu,
    nn.Dense(1),
])

generator = nn.Sequential([
    nn.Dense(8),
    nn.relu,
    nn.Dense(8),
    nn.relu,
    nn.Dense(4),
    nn.relu,
    nn.Dense(AMBIENT_DIM),
])
real_A = A.get_tensors()
real_B = B.get_tensors()
model = models.cyclegan.CycleGAN(generator, discriminator, (AMBIENT_DIM,), (AMBIENT_DIM,))
model.initialize(optax.sigmoid_binary_cross_entropy)
model.train(real_A, real_B, optax.adam(learning_rate=1e-3), print_every=5, batch_size=1000, num_epochs=700)