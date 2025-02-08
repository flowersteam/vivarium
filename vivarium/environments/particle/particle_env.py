import logging as lg

from functools import partial
from typing import Tuple

import jax.numpy as jnp

from jax import vmap, jit
from jax import random, ops

from jax_md import space, rigid_body, partition, quantity



from vivarium.environments.base_env import BaseEnv
from vivarium.environments.utils import normal, distance, relative_position
from vivarium.environments.physics_engine import (
    total_collision_energy,
    friction_force,
    dynamics_fn,
)
from vivarium.environments.particle import init_state
from vivarium.environments.particle.state import State



def disp(displacement, position, other_positions):
    return vmap(displacement, (None, 0))(position, other_positions)

def peak_f(x, mu, sigma):
  return jnp.exp(-((x-mu)/sigma)**2)

peak_f_vec = vmap(peak_f, (0, 0, 0))

peak_f_mat = vmap(peak_f_vec, (0, None, None))

# def lenia_energy_fn(displacement):
#     def lenia_energy(position, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep):
#         r = space.map_bond(displacement)(position, position)
#         U = peak_f_mat(r, mu_k, sigma_k).sum()*w_k
#         G = peak_f_vec(U, mu_g, sigma_g)
#         R = c_rep/2 * ((1.0-r).clip(0.0)**2).sum()
#         return R - G
#     return lenia_energy

def lenia_energy_fn(displacement):
    def lenia_energy(positions, x_position, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep):


        #r = jnp.linalg.norm(disp(displacement, x_position, positions), axis=1).clip(1e-10)
        #r = jnp.sqrt(jnp.square(x_position - positions).sum(-1).clip(1e-10))
        r = jnp.sqrt(jnp.square(disp(displacement, x_position, positions)).sum(-1).clip(1e-10))
        print('r', r)
        U = peak_f(r, mu_k, sigma_k).sum()*w_k
        G = peak_f(U, mu_g, sigma_g)
        R = c_rep/2 * ((1.0-r).clip(0.0)**2).sum()
        E = R - G
        #print('shapes', r, r.shape, U.shape, G.shape, R.shape, E.shape)
        return E
    return lenia_energy

def particle_lenia_force_fn(displacement):
    energy_fn = vmap(lenia_energy_fn(displacement), (None, 0, None, None, None, None, None, None))
    # def energy_fn(particle_state):
    #     return vmap_lenia_energy_fn(particle_state, particle_state.idx)
    def force_fn(state, neighbor, exists_mask):
       force = quantity.force(lambda x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep : lenia_energy_fn(displacement)(state.particle_state.position, x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep))
       return vmap(force)(state.particle_state.position, state.particle_state.mu_k, state.particle_state.sigma_k, state.particle_state.w_k, state.particle_state.mu_g, state.particle_state.sigma_g, state.particle_state.c_rep)
    #    return quantity.force(energy_fn)(state.particle_state.position, state.particle_state.position,
    #                                                           state.particle_state.mu_k,
    #                                                           state.particle_state.sigma_k,
    #                                                           state.particle_state.w_k,
    #                                                           state.particle_state.mu_g,
    #                                                           state.particle_state.sigma_g,
    #                                                           state.particle_state.c_rep)
    return force_fn

def dynamics_fn(displacement, shift, force_fn=None):
    """Compute the dynamics of the system

    :param displacement: displacement function of jax_md
    :param shift: shift function of jax_md
    :param force_fn: given force function, defaults to None
    :return: init_fn, step_fn functions of jax_md to compute the dynamics of the system
    """
    force_fn = force_fn(displacement) if force_fn else verlet_force_fn(displacement)

    def init_fn(state, key, kT=0.0):
        return state

    def step_fn(state, neighbor):
        """Compute the next state of the system

        :param state: current state of the system
        :param neighbor: neighbor array of the system
        :return: new state of the system
        """
        return state.particle_state.set(
            position=shift(state.particle_state.position, state.dt * force_fn(state, None, None))
        )

    return init_fn, step_fn


class ParticleEnv(BaseEnv):
    def __init__(self, state, seed=42):
        self.seed = seed
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.free()  #periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(
            self.displacement, self.shift, particle_lenia_force_fn
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
        # 1 : Compute agents proximeter
        exists_mask = jnp.where(state.particle_state.exists == 1, 1, 0)

        # 4 : Move the entities by applying forces on them (collision, friction and motor forces for agents)
        particle_state = self.apply_physics(state, neighbors)
        state = state.set(time=state.time + 1, particle_state=particle_state)

        # 5 : Update neighbors
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

    def allocate_neighbors(self, state, position=None):
        position = position or state.particle_state.position
        neighbors = self.neighbor_fn.allocate(position)
        return neighbors


if __name__ == "__main__":
    state = init_state(box_size=12., max_particles=200, dt=0.1)
    env = ParticleEnv(state)

    state_hist = []
    for _ in range(10000):
        state = env.step(state)
        state_hist.append(state)
        #print(state.particle_state.position[0])

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
        particles.set_data(state_hist[frame].particle_state.position[:, 0],
                           state_hist[frame].particle_state.position[:, 1])
        time_text.set_text(f'Time step: {state_hist[frame].time}')
        return particles, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(state_hist),
                                  init_func=init, blit=True)

    ani.save('particle_movements.mp4', writer='ffmpeg')
    plt.show()
