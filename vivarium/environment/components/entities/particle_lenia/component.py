import jax.numpy as jnp
from jax import vmap

from jax_md import quantity, partition
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.state import BaseParticleState
from vivarium.environment.components.entities import EntityComponent


from_mask_fn = lambda state: jnp.array(range(len(state.entity_state.entity_type_idx)))

to_mask_fn = lambda state: state.field(ParticleLeniaState).entity_idx


@md_dataclass
class ParticleLeniaState(BaseParticleState):
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    

def peak_f(x, mu, sigma):
  return jnp.exp(-((x - mu)/sigma)**2)


def lenia_energy_fn(displacement):
    def lenia_energy(x, positions, neighbors, neigh_mask, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep):
        target_positions = positions[neighbors]
        d_r = jnp.where(
            jnp.tile(neigh_mask, (2, 1)).T,
            vmap(displacement, (None, 0))(x, target_positions),
            0.)
        r = jnp.linalg.norm(d_r, axis=-1).clip(1e-10)
        p = jnp.where(neigh_mask, peak_f(r, mu_k, sigma_k), 0.0)
        s = p.sum()
        U = s * w_k
        G = peak_f(U, mu_g, sigma_g)
        R = jnp.where(neigh_mask,
                      c_rep/2 * ((1.0 - r).clip(0.0) ** 2),
                      0.0)
        R = R.sum()
        E = R - G
        return E
    return lenia_energy


def particle_lenia_state_fn(displacement, particle_lenia_state_field, from_mask_fn, to_mask_fn):
    
    def state_fn(state, neighbor, key):

        to_mask = to_mask_fn(state)

        particle_state = getattr(state, particle_lenia_state_field)
        neigh_mask = partition.neighbor_list_mask(neighbor, mask_self=True)
        force = quantity.force(
            lambda x, neigh, neigh_mask, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep: lenia_energy_fn(displacement)(
                x, state.entity_state.position, neigh, neigh_mask, mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep
                )
            )
        res = vmap(force)(
            state.entity_state.position[to_mask], 
            neighbor.idx[to_mask], neigh_mask[to_mask], 
            particle_state.mu_k, particle_state.sigma_k, particle_state.w_k, 
            particle_state.mu_g, particle_state.sigma_g, 
            particle_state.c_rep)
        all = jnp.zeros_like(state.entity_state.force)
        all = all.at[to_mask].set(res)
        return state.set(
            entity_state=state.entity_state.set(force=all + state.entity_state.force)
        )
    return state_fn


class ParticleLeniaComponent(EntityComponent):
    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction, exists,
                 mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep, subtype_labels=None
                 ):
        super().__init__(name=name, precedence=precedence, 
                         entity_type=entity_type, subtype=subtype,
                         position=position, orientation=orientation,
                         mass=mass, diameter=diameter, 
                         friction=friction, exists=exists,
                         subtype_labels=subtype_labels)  

        self.mu_k = jnp.array(mu_k)
        self.sigma_k = jnp.array(sigma_k)
        self.w_k = jnp.array(w_k)
        self.mu_g = jnp.array(mu_g)
        self.sigma_g = jnp.array(sigma_g)
        self.c_rep = jnp.array(c_rep)

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.entity_type] = ParticleLeniaState
        setattr(state_cls, self.entity_type, None)
        return state_cls
    

    def init_state_fn(self, state, neighbor_manager, key):
        entity_idx = jnp.arange(self.offset, self.offset + self.n_max, dtype=int)
        particle_state = state.__annotations__[self.entity_type](
                           entity_type=self.entity_type_int,
                           entity_idx=entity_idx,
                           mu_k=self.mu_k,
                           sigma_k=self.sigma_k,
                           w_k=self.w_k,
                           mu_g=self.mu_g,
                           sigma_g=self.sigma_g,
                           c_rep=self.c_rep,
        )

        return state.set(
            **{self.entity_type: particle_state}
        )

    def get_step_function(self, state, neighbor_manager, key):
        return particle_lenia_state_fn(neighbor_manager.displacement, 
                                       self.entity_type, 
                                       from_mask_fn, to_mask_fn)
      