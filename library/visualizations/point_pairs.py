from ast import Str
import plotly.express as px
from plotly import subplots, graph_objects as go
from jax import numpy as jnp

def visualize_point_correspondence(points_A: jnp.ndarray, points_B: jnp.ndarray, title_A: Str = 'points_A' , title_B: Str = 'points_B', showlegends: bool = False):

    assert (points_A.shape[-1] == 1 or points_A.shape[-1] == 2 or points_A.shape[-1] == 3) and \
            (points_B.shape[-1] == 1 or points_B.shape[-1] == 2 or points_B.shape[-1] == 3), \
            "Only 1-, 2- or 3- dimensional visualizations are supported!"

    types = [None, None]

    types[0] = 'xy' if points_A.shape[-1] <= 2 else 'scene'
    types[1] = 'xy' if points_B.shape[-1] <= 2 else 'scene'

    fig = subplots.make_subplots(rows=1, cols=2, specs=[
                                    [{'type': item} for item in types]],
                                    subplot_titles=(title_A, title_B))

    def add_scatter(points: jnp.ndarray, levels: jnp.ndarray, row: int, col: int):
        data = {
            'mode': 'markers',
            'marker': {
                'size': 2.5,
                'color': levels
            },
        }
        
        if points.shape[-1] == 1:
            data.update({
                'x': points[:, 1], 'y': jnp.zeros(points.shape[0])
            })
            
            fig.add_trace(go.Scatter(data, showlegend=showlegends), row=row, col=col)

            fig.update_xaxes(title_text='x', row=row, col=col)
            fig.update_yaxes(title_text='This axis is not used.', row=row, col=col)
            
        elif points.shape[-1] == 2:
            data.update({
                'x': points[:, 0], 'y': points[:, 1],
            })
            
            fig.add_trace(go.Scatter(data, showlegend=showlegends), row=row, col=col)
            
            fig.update_xaxes(title_text='x', row=row, col=col)
            fig.update_yaxes(title_text='y', row=row, col=col)
            
        else:
            data.update(
                {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
            )
            
            fig.add_trace(go.Scatter3d(data,showlegend=showlegends), row=row, col=col)
            
            fig.update_xaxes(title_text='x_0', row=row, col=col)
            fig.update_yaxes(title_text='x_1', row=row, col=col)

    levels = jnp.sqrt(
        jnp.sum(
            jnp.square(points_A),
            axis=-1))

    add_scatter(points_A, levels, 1, 1)
    add_scatter(points_B, levels, 1, 2)

    return fig