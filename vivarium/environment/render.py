import time

import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output

from vivarium.environment.utils import normal

def plot_particles(ax, state, type, color, size_scale=30):
    entities = getattr(state, type)
    idx = entities.entity_idx
    
    exists = state.entity_state.exists[idx]         
    exists = jnp.where(exists != 0)
    pos = state.entity_state.position_center[idx][exists]
    diameter = state.entity_state.diameter[idx][exists][exists]
    x, y = pos[:, 0], pos[:, 1]

    colors = [color] * state.entity_state.exists[state.e_cond(type)].sum().item()

    ax.scatter(
        x,
        y,
        c=colors,
        s=diameter * size_scale,
        label=type
    )


def plot_orientation(ax, state, type, color, arrow_length):
    entities = getattr(state, type)
    idx = entities.entity_idx
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
    colors = [color] * state.entity_state.exists[state.e_cond(type)].sum().item()
    ax.quiver(
        x,
        y,
        dx,
        dy,
        color=colors,
        scale=1,
        scale_units="xy",
        headwidth=0.8,
        angles="xy",
        width=0.01,
    )

# Functions to render the current state
def render(state, box_size, agent_field='agents', object_field='objects', colors={'agents': 'red', 'objects': 'blue'}):
    
    plt.figure(figsize=(6, 6))
    plt.xlim(0, box_size)
    plt.xlim(0, box_size)

    arrow_length = 3

    if agent_field in state.__dataclass_fields__:
        plot_particles(plt, state, agent_field, colors[agent_field])
        plot_orientation(plt, state, agent_field, colors[agent_field], arrow_length)

    if object_field in  state.__dataclass_fields__:
        plot_particles(plt, state, object_field, colors[object_field])

    plt.title("State")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    plt.show()

# Function to render a state history
def render_history(state_history, box_size, agent_field='agents', object_field='objects', colors={'agents': 'red', 'objects': 'blue'}, fps=10, skip_frames=1, arrow_length=3, filename=None):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    def update(t):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

        if agent_field in state_history[t].__dataclass_fields__:
            plot_particles(ax, state_history[t], agent_field, color=colors[agent_field])
            plot_orientation(ax, state_history[t], agent_field, color=colors[agent_field], arrow_length=arrow_length)

        if object_field in state_history[t].__dataclass_fields__:
            plot_particles(ax, state_history[t], object_field, color=colors[object_field])

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
