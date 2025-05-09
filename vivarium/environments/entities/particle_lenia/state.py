import logging as lg

import jax.numpy as jnp
from jax import vmap

from jax_md import quantity
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import ParticleState

from_mask_fn = lambda state: jnp.array(range(len(state.entity_state.entity_type_idx)))

to_mask_fn = lambda state: state.field(ParticleLeniaState).entity_idx


def get_state_function(state, neighbor_manager):
    return particle_lenia_state_fn(neighbor_manager.displacement, state.field_name(ParticleLeniaState), from_mask_fn, to_mask_fn)


@md_dataclass
class ParticleLeniaState(ParticleState):
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type)
    
    def state_fns(self):
        return [get_state_function]
    

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


def particle_lenia_state_fn(displacement, particle_lenia_state_field, from_mask_fn, to_mask_fn):
    
    def state_fn(state, neighbor):
        from_mask = from_mask_fn(state)
        to_mask = to_mask_fn(state)
        force = quantity.force(
            lambda x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep : lenia_energy_fn(displacement)(state.entity_state.position[from_mask], x, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep)
            )
        particle_state = getattr(state, particle_lenia_state_field)
        res = vmap(force)(state.entity_state.position[to_mask], particle_state.mu_k, particle_state.sigma_k, particle_state.w_k, particle_state.mu_g, particle_state.sigma_g, particle_state.c_rep)
        all = jnp.zeros_like(state.entity_state.force)
        all = all.at[to_mask].set(res)
        return state.set(
            entity_state=state.entity_state.set(force=all + state.entity_state.force)
        )
    return state_fn
