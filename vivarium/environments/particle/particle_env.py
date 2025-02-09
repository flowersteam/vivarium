import logging as lg

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from jax import vmap, jit
from jax import random, ops

from jax_md import space, simulate, partition, quantity



from vivarium.environments.base_env import BaseEnv
from vivarium.environments.utils import normal, distance, relative_position
from vivarium.environments.physics_engine import (
    total_collision_energy,
    # friction_force,
    dynamics_fn,
)
from vivarium.environments.particle import init_state, LENIA_PARAMS, N_DIMS
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

def collision_force(displacement):
    coll_force_fn = quantity.force(
        partial(total_collision_energy, displacement=displacement)
    )

    def force_fn(state, neighbor, exists_mask):
        """Returns the collision force function of the environment

        :param state: state
        :param neighbor: neighbor maps of entities
        :param exists_mask: mask on existing entities
        :return: collision force function
        """
        return coll_force_fn(
            state.entities.position,
            neighbor=neighbor,
            exists_mask=exists_mask,
            diameter=state.entities.diameter,
            epsilon=state.collision_eps,
            alpha=state.collision_alpha,
        )
    return force_fn

def friction_force(state, neighbor, exists_mask):
    """Compute the friction force on the system

    :param state: current state of the system
    :param exists_mask: mask to specify which particles exist
    :return: friction force on the system
    """
    cur_vel = state.entities.momentum / state.entities.mass
    # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities)
    mask = jnp.stack([exists_mask] * 2, axis=1)
    cur_vel = jnp.where(mask, cur_vel, 0.0)
    return -jnp.tile(state.particle_state.friction, (N_DIMS, 1)).T * cur_vel

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

# def force_fn(displacement):
#     lenia = particle_lenia_force_fn(displacement)
#     collision = collision_force(displacement)
#     def fn(state, neighbor, exists_mask):
#         force = lenia(state, neighbor, exists_mask)
#         force += collision(state, neighbor, exists_mask)
#         force += friction_force(state, exists_mask)
#         return force
#     return fn

def force_fn(displacement, force_fns):
    fns = [fn(displacement) for fn in force_fns]
    def force(state, neighbor, exists_mask):
        return jnp.array([fn(state, neighbor, exists_mask) for fn in fns]).sum(0)
    return force

def dynamics_fn(displacement, shift, force_fn=None):
    """Compute the dynamics of the system

    :param displacement: displacement function of jax_md
    :param shift: shift function of jax_md
    :param force_fn: given force function, defaults to None
    :return: init_fn, step_fn functions of jax_md to compute the dynamics of the system
    """
    force_fn = force_fn(displacement) if force_fn else verlet_force_fn(displacement)

    def init_fn(state, key, kT=0.0):
        key, _ = jax.random.split(key)
        assert state.particle_state.momentum is None
        assert not jnp.any(state.particle_state.force)

        state = state.set(particle_state=simulate.initialize_momenta(state.particle_state, key, kT))
        return state

    def mask_momentum(entity_state, exists_mask):
        """
        Set the momentum values to zeros for non existing entities
        :param entity_state: entity_state
        :param exists_mask: bool array specifying which entities exist or not
        :return: entity_state: new entities state state with masked momentum values
        """

        exists_mask = jnp.stack([exists_mask] * N_DIMS, axis=1)
        momentum = jnp.where(exists_mask, entity_state.momentum, 0)
        return entity_state.set(momentum=momentum)

    def step_fn(state, neighbor):
        """Compute the next state of the system

        :param state: current state of the system
        :param neighbor: neighbor array of the system
        :return: new state of the system
        """
        # return state.particle_state.set(
        #     position=shift(state.particle_state.position, state.dt * force_fn(state, None, None))
        # )

        exists_mask = (
            state.particle_state.exists == 1
        )  # Only existing entities have effect on others
        dt_2 = state.dt / 2.0
        # Compute forces
        force = force_fn(state, neighbor, exists_mask)
        # Compute changes on entities
        entity_state = simulate.momentum_step(state.particle_state, dt_2)
        # TODO : why do we used dt and not dt/2 in the line below ?
        entity_state = simulate.position_step(
            entity_state, shift, dt_2, neighbor=neighbor
        )
        entity_state = entity_state.set(force=force)
        entity_state = simulate.momentum_step(entity_state, dt_2)
        entity_state = mask_momentum(entity_state, exists_mask)
        return entity_state

    return init_fn, step_fn


class ParticleEnv(BaseEnv):
    def __init__(self, state, seed=42):
        self.seed = seed
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.free()  #periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(
            self.displacement, self.shift, 
            partial(force_fn, force_fns=[particle_lenia_force_fn,
                                         collision_force,
                                         lambda _: friction_force])
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

        # exists_mask = jnp.where(state.particle_state.exists == 1, 1, 0)

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
    std = 0.
    params = {param: (mean, std) for param, mean in LENIA_PARAMS.items()}
    params['c_rep'] = 0.
    state = init_state(box_size=12., max_particles=800, dt=0.1, friction=0.2, **params)
    env = ParticleEnv(state)

    state_hist = []
    n_steps  = 1000
    for _ in range(n_steps):
        state = env.step(state)
        state_hist.append(state)
        #print(state.particle_state.position[0])


    # def step_f(state, _):
    #     state, _ = env._step(state, env.neighbors)
    #     return state, state.particle_state.position
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
        particles.set_data(state_hist[frame].particle_state.position[:, 0],
                           state_hist[frame].particle_state.position[:, 1])
        time_text.set_text(f'Time step: {frame}')
        return particles, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(state_hist),
                                  init_func=init, blit=True)

    ani.save('particle_movements.mp4', writer='ffmpeg')
    plt.show()
