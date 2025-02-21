import logging as lg

import jax.numpy as jnp

from jax import vmap
from jax import random

from jax_md import space, quantity

from vivarium.environments.base_env import BaseEnv, NeighborManager
from vivarium.environments.utils import distance
from vivarium.environments.physics_engine import (
    reset_force_state_fn,
    collision_state_fn,
    friction_state_fn,
    init_state_fn,
    step_state_fn
)
from vivarium.environments.particle_lenia.state import init_state, LENIA_PARAMS


def disp(displacement, position, other_positions):
    return vmap(displacement, (None, 0))(position, other_positions)


def peak_f(x, mu, sigma):
  return jnp.exp(-((x - mu)/sigma)**2)


def lenia_energy_fn(displacement):
    def lenia_energy(positions, x_position, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep):
        r = jnp.sqrt(jnp.square(disp(displacement, x_position, positions)).sum(-1).clip(1e-10))
        U = peak_f(r, mu_k, sigma_k).sum() * w_k
        G = peak_f(U, mu_g, sigma_g)
        R = c_rep/2 * ((1.0 - r).clip(0.0) ** 2).sum()
        E = R - G
        return E
    return lenia_energy


from_mask_fn = lambda state: jnp.array(range(len(state.entities.entity_idx)))

to_mask_fn = lambda state: state.objects.ent_idx


def particle_lenia_state_fn(displacement, from_mask_fn=from_mask_fn, to_mask_fn=to_mask_fn):
    # energy_fn = vmap(lenia_energy_fn(displacement), (None, 0, None, None, None, None, None, None))
    
    def state_fn(state, neighbor):
        from_mask = from_mask_fn(state)
        to_mask = to_mask_fn(state)
        force = quantity.force(
            lambda x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep : lenia_energy_fn(displacement)(state.entities.position[from_mask], x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep)
            )
        res = vmap(force)(state.entities.position[to_mask], state.objects.mu_k, state.objects.sigma_k, state.objects.w_k, state.objects.mu_g, state.objects.sigma_g, state.objects.c_rep)
        all = jnp.zeros_like(state.entities.force)
        all = all.at[to_mask].set(res)
        return state.set(
            entities=state.entities.set(force=all + state.entities.force)
        )
    return state_fn


class ParticleLeniaEnv(BaseEnv):
    def __init__(self, state, space_fn=space.periodic, seed=42):
        
        displacement, shift = space_fn(state.box_size)

        exists_mask_fn = lambda state: state.entities.exists == 1
        key = random.PRNGKey(seed)
        key, new_key = random.split(key)
        init_fn = init_state_fn(key)
        neighbor_manager = NeighborManager(displacement, state)
        state_fns = [reset_force_state_fn(),
                     particle_lenia_state_fn(displacement, from_mask_fn, to_mask_fn),
                     collision_state_fn(displacement, exists_mask_fn),
                     friction_state_fn(exists_mask_fn),
                     step_state_fn(shift, exists_mask_fn, new_key)]
        super().__init__(state, init_fn, state_fns, neighbor_manager)


if __name__ == "__main__":
    std = 0.
    params = {param: (mean, std) for param, mean in LENIA_PARAMS.items()}
    params['c_rep'] = 0.
    n_particles = 200
    state = init_state(box_size=100, max_particles=n_particles, dt=0.1, friction=1., **params)

    pos_range = 12.

    key = random.PRNGKey(42)
    key, new_key = random.split(key)

    state = state.set(
        entities=state.entities.set(
            position = state.entities.position.at[state.objects.ent_idx].set(
                random.uniform(new_key, (n_particles, 2)) * pos_range + state.box_size / 2 - pos_range / 2  * jnp.ones((n_particles, 2))
                )
        )
    )

    env = ParticleLeniaEnv(state)

    state_hist = []
    n_steps  = 100
    for _ in range(n_steps):
        state = env.step(state, num_scan_steps=10)
        state_hist.append(state)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    particles, = ax.plot([], [], 'bo', ms=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.set_xlim(0, state.box_size)
        ax.set_ylim(0, state.box_size)
        time_text.set_text('')
        return particles, time_text

    def update(frame):
        particles.set_data(state_hist[frame].entities.position[:, 0],
                           state_hist[frame].entities.position[:, 1])
        time_text.set_text(f'Time step: {frame}')
        return particles, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(state_hist),
                                  init_func=init, blit=True)

    ani.save('particle_movements.mp4', writer='ffmpeg')
    plt.show()
