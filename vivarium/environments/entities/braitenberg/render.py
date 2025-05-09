import time
from IPython.display import display, clear_output

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from vivarium.environments.utils import normal

import matplotlib.animation as animation


def plot_particles(ax, state, type, size_scale=30):
    entities = getattr(state, type)
    idx = entities.ent_idx
    
    exists = state.entity_state.exists[idx]         
    exists = jnp.where(exists != 0)
    pos = state.entity_state.position_center[idx][exists]

    diameter = state.entity_state.diameter[idx][exists][exists]
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
    exists = state.entity_state.exists[idx]         
    exists = jnp.where(exists != 0)

    pos = state.entity_state.position_center[idx][exists]
    x, y = pos[:, 0], pos[:, 1]

    theta = state.entity_state.position_orientation[idx][exists][
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

    if hasattr(state, 'agent_state'):
        plot_particles(plt, state, 'agent_state')
        plot_orientation(plt, state, 'agent_state', arrow_length)

    if hasattr(state, 'object_state'):
        plot_particles(plt, state, 'object_state')

    plt.title("State")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    plt.show()

# Function to render a state history
def render_history(state_history, fps=10, skip_frames=1, arrow_length=3, filename=None):
    box_size = state_history[0].box_size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    def update(t):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

        if hasattr(state_history[t], 'agent_state'):
            plot_particles(ax, state_history[t], 'agent_state')
            plot_orientation(ax, state_history[t], 'agent_state', arrow_length)

        if hasattr(state_history[t], 'object_state'):
            plot_particles(ax, state_history[t], 'object_state')

        ax.set_title(f"Timestep: {t}")
        # ax.set_xlabel("X Position")
        # ax.set_ylabel("Y Position")
        ax.axis('off')

    if filename:
        ani = animation.FuncAnimation(fig, update, frames=range(0, len(state_history), skip_frames))
        ani.save(filename, writer='ffmpeg', fps=fps)
    else:
        for t in range(0, len(state_history), skip_frames):
            update(t)
            display(fig)
            clear_output(wait=True)
            time.sleep(1/fps)

    plt.close(fig)
