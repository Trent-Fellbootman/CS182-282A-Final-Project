from library import datasets, models
from flax import linen as nn
from jax import random
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
taylor_pair_dis = datasets.point_dataset.PointPairDistribution.random_taylor(
latent_dim=2,
dis_A_dim=3,
dis_B_dim=3,
latent_range=1,
max_order=2,
coeff_range=1,
noise_std_A=0.05,
noise_std_B=0,
key=random.PRNGKey(2))

dataset = taylor_pair_dis.generate_dataset(1000)
A, B = dataset.get_all_point_pairs()
discriminator = nn.Sequential([
    nn.Dense(16),
    nn.relu,
    nn.Dense(32),
    nn.relu,
    nn.Dense(1),
    nn.sigmoid
])

generator = nn.Sequential([
    nn.Dense(16),
    nn.relu,
    nn.Dense(32),
    nn.relu,
    nn.Dense(3)
])

model = models.cyclegan.CycleGAN(generator, discriminator, A.shape, B.shape)
model.initialize()
model.train(A, B, 20)