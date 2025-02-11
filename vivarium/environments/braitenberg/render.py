import time
from IPython.display import display, clear_output

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from vivarium.environments.utils import normal


def _string_to_rgb(color_str):
    return jnp.array(list(colors.to_rgb(color_str)))


# Functions to render the current state
def render(state):
    box_size = state.box_size
    max_agents = state.max_agents

    plt.figure(figsize=(6, 6))
    plt.xlim(0, box_size)
    plt.xlim(0, box_size)

    exists_agents, exists_objects = (
        state.entities.exists[:max_agents],
        state.entities.exists[max_agents:],
    )
    exists_agents = jnp.where(exists_agents != 0)
    exists_objects = jnp.where(exists_objects != 0)

    agents_pos = state.entities.unified_position[:max_agents][exists_agents]
    agents_theta = state.entities.unified_orientation[:max_agents][exists_agents][
        exists_agents
    ]
    agents_diameter = state.entities.diameter[:max_agents][exists_agents][exists_agents]
    objects_pos = state.entities.unified_position[max_agents:][exists_objects]
    object_diameter = state.entities.diameter[max_agents:][exists_objects]

    x_agents, y_agents = agents_pos[:, 0], agents_pos[:, 1]
    agents_colors_rgba = [
        colors.to_rgba(np.array(c), alpha=1.0)
        for c in state.agents.color[exists_agents]
    ]
    x_objects, y_objects = objects_pos[:, 0], objects_pos[:, 1]
    object_colors_rgba = [
        colors.to_rgba(np.array(c), alpha=1.0)
        for c in state.objects.color[exists_objects]
    ]

    n = normal(agents_theta)

    arrow_length = 3
    size_scale = 30
    dx = arrow_length * n[:, 0]
    dy = arrow_length * n[:, 1]
    plt.quiver(
        x_agents,
        y_agents,
        dx,
        dy,
        color=agents_colors_rgba,
        scale=1,
        scale_units="xy",
        headwidth=0.8,
        angles="xy",
        width=0.01,
    )
    plt.scatter(
        x_agents,
        y_agents,
        c=agents_colors_rgba,
        s=agents_diameter * size_scale,
        label="agents",
    )
    plt.scatter(
        x_objects,
        y_objects,
        c=object_colors_rgba,
        s=object_diameter * size_scale,
        label="objects",
    )

    plt.title("State")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    plt.show()


# Function to render a state hystory
def render_history(state_history, pause=0.001, skip_frames=1):
    box_size = state_history[0].box_size
    max_agents = state_history[0].max_agents
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    for t in range(0, len(state_history), skip_frames):
        # Because weird saving at the moment, we don't save the state but all its sub-elements
        entities = state_history[t].entities
        agents = state_history[t].agents
        objects = state_history[t].objects

        exists_agents, exists_objects = (
            entities.exists[:max_agents],
            entities.exists[max_agents:],
        )
        exists_agents = jnp.where(exists_agents != 0)
        exists_objects = jnp.where(exists_objects != 0)

        agents_pos = entities.unified_position[:max_agents][exists_agents]
        agents_theta = entities.unified_orientation[:max_agents][exists_agents][
            exists_agents
        ]
        agents_diameter = entities.diameter[:max_agents][exists_agents][exists_agents]
        objects_pos = entities.unified_position[max_agents:][exists_objects]
        object_diameter = entities.diameter[max_agents:][exists_objects]

        x_agents, y_agents = agents_pos[:, 0], agents_pos[:, 1]
        agents_colors_rgba = [
            colors.to_rgba(np.array(c), alpha=1.0) for c in agents.color[exists_agents]
        ]
        x_objects, y_objects = objects_pos[:, 0], objects_pos[:, 1]
        object_colors_rgba = [
            colors.to_rgba(np.array(c), alpha=1.0)
            for c in objects.color[exists_objects]
        ]

        n = normal(agents_theta)

        arrow_length = 3
        size_scale = 30
        dx = arrow_length * n[:, 0]
        dy = arrow_length * n[:, 1]

        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

        ax.quiver(
            x_agents,
            y_agents,
            dx,
            dy,
            color=agents_colors_rgba,
            scale=1,
            scale_units="xy",
            headwidth=0.8,
            angles="xy",
            width=0.01,
        )
        ax.scatter(
            x_agents,
            y_agents,
            c=agents_colors_rgba,
            s=agents_diameter * size_scale,
            label="agents",
        )
        ax.scatter(
            x_objects,
            y_objects,
            c=object_colors_rgba,
            s=object_diameter * size_scale,
            label="objects",
        )

        ax.set_title(f"Timestep: {t}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()

        display(fig)
        clear_output(wait=True)
        time.sleep(pause)

    plt.close(fig)
