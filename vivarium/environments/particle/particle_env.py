import logging as lg

from functools import partial

import jax.numpy as jnp

from jax import vmap, jit
from jax import random

from jax_md import space, partition, quantity



from vivarium.environments.base_env import BaseEnv
from vivarium.environments.utils import distance
from vivarium.environments.physics_engine import (
    collision_force_fn,
    friction_force_fn,
    sum_force_fns,
    dynamics_fn,
)
from vivarium.environments.particle import init_state, LENIA_PARAMS
from vivarium.environments.particle.state import State



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

def particle_lenia_force_fn(displacement):
    # energy_fn = vmap(lenia_energy_fn(displacement), (None, 0, None, None, None, None, None, None))
    
    def force_fn(state, neighbor, exists_mask):
       force = quantity.force(lambda x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep : lenia_energy_fn(displacement)(state.entities.position, x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep))
       return vmap(force)(state.entities.position, state.objects.mu_k, state.objects.sigma_k, state.objects.w_k, state.objects.mu_g, state.objects.sigma_g, state.objects.c_rep)
    return force_fn


class ParticleEnv(BaseEnv):
    def __init__(self, state, seed=42):
        self.seed = seed
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.free()  #periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(
            self.displacement, self.shift, 
            sum_force_fns(self.displacement, force_fns=[particle_lenia_force_fn,
                                                        collision_force_fn,
                                                        friction_force_fn])
        )
        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            state.box_size,
            r_cutoff=state.neighbor_radius,
            dr_threshold=10.0,
            capacity_multiplier=1.5,
            format=partition.Sparse,
        )

        self.neighbors = self.allocate_neighbors(state)

    def distance(self, point1, point2):
        return distance(self.displacement, point1, point2)

    @partial(jit, static_argnums=(0,))
    def _step(
        self, state, neighbors: jnp.array
    ):

        entities = self.apply_physics(state, neighbors)
        state = state.set(time=state.time + 1, entities=entities)

        neighbors = neighbors.update(state.entities.position)
        return state, neighbors

    def step(self, state: State) -> State:
        if state.entities.momentum is None:
            state = self.init_fn(state, self.init_key)
        current_state = state
        state, neighbors = self._step(
            current_state, self.neighbors
        )
        if self.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state
            lg.warning(
                f"NEIGHBORS BUFFER OVERFLOW at step {state.time}: rebuilding neighbors"
            )
            neighbors, self.agents_neighs_idx = self.allocate_neighbors(state)
            assert not neighbors.did_buffer_overflow

        self.neighbors = neighbors
        return state


if __name__ == "__main__":
    std = 0.
    params = {param: (mean, std) for param, mean in LENIA_PARAMS.items()}
    params['c_rep'] = 0.
    state = init_state(box_size=12., max_particles=200, dt=0.1, friction=0.25, **params)
    env = ParticleEnv(state)

    state_hist = []
    n_steps  = 100
    for _ in range(n_steps):
        state = env.step(state)
        state_hist.append(state)


    # def step_f(state, _):
    #     state, _ = env._step(state, env.neighbors)
    #     return state, state.entities.position
    # state_hist = jax.lax.scan(step_f, state, None, n_steps)[1]

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    particles, = ax.plot([], [], 'bo', ms=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.set_xlim(- state.box_size, state.box_size * 2)
        ax.set_ylim(- state.box_size, state.box_size * 2)
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
