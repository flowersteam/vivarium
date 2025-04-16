import logging as lg

import jax.numpy as jnp
from jax import random
from jax import vmap

from jax_md import space, quantity

from vivarium.environments.base_env import BaseEnv, NeighborManager
from vivarium.environments.physics_engine import (
    reset_force_state_fn,
    collision_state_fn,
    friction_state_fn,
    init_state_fn,
    step_state_fn
)

from vivarium.environments.particle_lenia.state import ParticleLeniaState


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


from_mask_fn = lambda state: jnp.array(range(len(state.entity_state.entity_type_idx)))

to_mask_fn = lambda state: state.field(ParticleLeniaState).entity_idx


def particle_lenia_state_fn(displacement, from_mask_fn=from_mask_fn, to_mask_fn=to_mask_fn):
    
    def state_fn(state, neighbor):
        from_mask = from_mask_fn(state)
        to_mask = to_mask_fn(state)
        force = quantity.force(
            lambda x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep : lenia_energy_fn(displacement)(state.entity_state.position[from_mask], x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep)
            )
        particle_state = state.field(ParticleLeniaState)
        res = vmap(force)(state.entity_state.position[to_mask], particle_state.mu_k, particle_state.sigma_k, particle_state.w_k, particle_state.mu_g, particle_state.sigma_g, particle_state.c_rep)
        all = jnp.zeros_like(state.entity_state.force)
        all = all.at[to_mask].set(res)
        return state.set(
            entity_state=state.entity_state.set(force=all + state.entity_state.force)
        )
    return state_fn


class ParticleLeniaEnv(BaseEnv):
    def __init__(self, state, box_size, neighbor_radius, seed=42, space_fn=space.periodic, **kwargs):
        
        displacement, shift = space_fn(box_size)

        exists_mask_fn = lambda state: state.entity_state.exists == 1
        key = random.PRNGKey(seed)
        key, new_key = random.split(key)
        init_fn = init_state_fn(key)
        neighbor_manager = NeighborManager(displacement, box_size, neighbor_radius, state)
        state_fns = [reset_force_state_fn(),
                     particle_lenia_state_fn(displacement, from_mask_fn, to_mask_fn),
                     collision_state_fn(displacement, exists_mask_fn),
                     friction_state_fn(exists_mask_fn),
                     step_state_fn(shift, exists_mask_fn, new_key)]
        super().__init__(state, init_fn, state_fns, neighbor_manager, **kwargs)
