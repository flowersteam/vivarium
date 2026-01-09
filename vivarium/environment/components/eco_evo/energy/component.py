import jax
import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.component import Component
from vivarium.environment.utils import type_mask


@md_dataclass
class EnergyState:
    energy: jnp.ndarray = None
    energy_init: jnp.ndarray = None
    energy_max: jnp.ndarray = None
    energy_burst: jnp.ndarray = None
    energy_decay: jnp.ndarray = None
            

class EnergyComponent(Component):
    def __init__(self, name, precedence,
                 entity_type,
                 energy_init, energy_max, energy_decay, 
                 energy_burst):
        super().__init__(name, precedence)
        
        self.energy_init = jnp.array(energy_init)
        self.energy_max = jnp.array(energy_max)
        self.energy_burst = jnp.array(energy_burst)
        self.energy_decay = jnp.array(energy_decay)
        self.entity_type = entity_type
        self.state_attr = f'{self.name}_state'

    def get_step_function(self, state, neighbor_manager, key):

        n_entities = state.entity_state.exists.shape[0]

        def state_fn(state, neighbors, key):

            energy_state = getattr(state, self.state_attr)
            
            target_mask = type_mask(state.entity_state)           
            
            consumption_matrix = state.consumption_state.consumption_matrix

            # # `consuming` is the number of other entities consumed by each entity.
            # # It is a float, so that when multiple entities consume the same target,
            # # they only get the corresponding fraction of it.
            consuming = consumption_matrix.sum(axis=1)
            
            neigh_flat = neighbors.idx.ravel()
            consumption_matrix_flat = consumption_matrix.ravel()
            
            # Similar logic as in consuming
            consumed = jax.ops.segment_sum(consumption_matrix_flat, neigh_flat, n_entities)            
            
            energy = energy_state.energy + (consuming - consumed) * energy_state.energy_burst
            
            energy = jnp.where(
                target_mask,
                energy - energy_state.energy_decay,
                energy
            )

            # Redistribute the extra energy (positive or negative)
            # from entities outside bounds to those within bounds
            energy_outside_bounds = (energy <= 0) | (energy > energy_state.energy_max)
            bounds = jnp.zeros_like(energy)
            bounds = jnp.where(energy > energy_state.energy_max, energy_state.energy_max, bounds)            
            extra_energy = jnp.where(target_mask & energy_outside_bounds, energy - bounds, 0.).sum()
            per_entity = extra_energy / jnp.sum(target_mask & ~energy_outside_bounds)             
            energy = jnp.where(
                target_mask & ~energy_outside_bounds,
                energy + per_entity,
                energy
            )
            energy = jnp.where(
                target_mask & energy_outside_bounds,
                jnp.clip(energy, 0, energy_state.energy_max),
                energy,
            )
            
            return state.set(
                    **{self.state_attr: getattr(state, self.state_attr).set(
                        energy=energy
                    )}
                )

        return state_fn

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.state_attr] = EnergyState
        setattr(state_cls, self.state_attr, None)    
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        if len(self.energy_init.shape) == 0:
            energy = jnp.full(state.entity_state.count(), self.energy_init)
        else:
            energy = self.energy_init
        return state.set(
            **{self.state_attr: EnergyState(
                energy=energy,
                energy_init=self.energy_init,
                energy_max=self.energy_max,
                energy_burst=self.energy_burst,
                energy_decay=self.energy_decay,  
            )})
        