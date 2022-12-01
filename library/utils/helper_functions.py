from jax import random, numpy as jnp
from flax import linen as nn
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def check_model_forward(model: nn.Module, input_data: jnp.array):
    key = random.PRNGKey(0) # DO NOT CHANGE THIS
    test_input = random.normal(key, input_data.shape)
    params = model.init(key, test_input)
    return model.apply(params, test_input)


def plot_generated_samples(real_A, real_B, fake_A, fake_B):
    fig = make_subplots(rows=1, cols=2, subplot_titles = ["Distribution A", "Distribution B"])
    fig.add_trace(go.Scatter(x=real_A[:, 0], y=real_A[:, 1], mode='markers', marker=dict(color="blue"),name='Real A'), row=1, col=1)
    fig.add_trace(go.Scatter(x=real_B[:, 0], y=real_B[:, 1], mode='markers', marker=dict(color="green"),name='Real B'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fake_A[:, 0], y=fake_A[:, 1], mode='markers', marker=dict(color="red"),name='Fake'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fake_B[:, 0], y=fake_B[:, 1], mode='markers', marker=dict(color="red"),name='Fake', showlegend=False), row=1, col=2)
    fig.update_layout(title_text="Real samples and fake samples from the trained generators")
    fig.show()

def plot_samples(real_A, real_B):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution A", "Distribution B"])
    fig.add_trace(go.Scatter(x=real_A[:, 0], y=real_A[:, 1], mode='markers', marker=dict(color="blue"),name='Real A'), row=1, col=1)
    fig.add_trace(go.Scatter(x=real_B[:, 0], y=real_B[:, 1], mode='markers', marker=dict(color="green"), name='Real B'), row=1, col=2)
    fig.update_layout(title_text="Real samples")
    fig.show()