import time
from IPython.display import display, clear_output

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from vivarium.environments.utils import normal


def _string_to_rgb(color_str):
    return jnp.array(list(colors.to_rgb(color_str)))


def plot_particles(ax, state, type, size_scale=30):
    entities = getattr(state, type)
    idx = entities.ent_idx
    
    exists = state.entities.exists[idx]         
    exists = jnp.where(exists != 0)
    pos = state.entities.unified_position[idx][exists]

    diameter = state.entities.diameter[idx][exists][exists]
    x, y = pos[:, 0], pos[:, 1]
    colors_rgba = [
        colors.to_rgba(np.array(c), alpha=1.0) for c in entities.color[exists]
    ]

    ax.scatter(
        x,
        y,
        c=colors_rgba,
        s=diameter * size_scale,
        label=type
    )

def plot_orientation(ax, state, type, arrow_length):
    entities = getattr(state, type)
    idx = entities.ent_idx
    exists = state.entities.exists[idx]         
    exists = jnp.where(exists != 0)

    pos = state.entities.unified_position[idx][exists]
    x, y = pos[:, 0], pos[:, 1]

    theta = state.entities.unified_orientation[idx][exists][
        exists
    ]
    n = normal(theta)
    
    dx = arrow_length * n[:, 0]
    dy = arrow_length * n[:, 1]
    colors_rgba = [
        colors.to_rgba(np.array(c), alpha=1.0) for c in entities.color[exists]
    ]
    ax.quiver(
        x,
        y,
        dx,
        dy,
        color=colors_rgba,
        scale=1,
        scale_units="xy",
        headwidth=0.8,
        angles="xy",
        width=0.01,
    )

# Functions to render the current state
def render(state):
    box_size = state.box_size
    max_agents = state.max_agents

    plt.figure(figsize=(6, 6))
    plt.xlim(0, box_size)
    plt.xlim(0, box_size)

    arrow_length = 3
    # size_scale = 30

    if hasattr(state, 'agents'):
        plot_particles(plt, state, 'agents')
        plot_orientation(plt, state, 'agents', arrow_length)

    if hasattr(state, 'objects'):
        plot_particles(plt, state, 'objects')

    plt.title("State")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    plt.show()

import matplotlib.animation as animation

# Function to render a state history
def render_history(state_history, pause=0.001, skip_frames=1, arrow_length=3, filename=None):
    box_size = state_history[0].box_size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    def update(t):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

        if hasattr(state_history[t], 'agents'):
            plot_particles(ax, state_history[t], 'agents')
            plot_orientation(ax, state_history[t], 'agents', arrow_length)

        if hasattr(state_history[t], 'objects'):
            plot_particles(ax, state_history[t], 'objects')

        ax.set_title(f"Timestep: {t}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    if filename:
        ani = animation.FuncAnimation(fig, update, frames=range(0, len(state_history), skip_frames))
        ani.save(filename, writer='ffmpeg', fps=1/pause)
    else:
        for t in range(0, len(state_history), skip_frames):
            update(t)
            display(fig)
            clear_output(wait=True)
            time.sleep(pause)

    plt.close(fig)
