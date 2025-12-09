import jax
import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.component import Component
from vivarium.environment.utils import type_mask


class EnergyComponent(Component):
    def __init__(self, name, precedence,
                 entity_type, subtype,
                 energy_init, energy_max, energy_decay, 
                 energy_burst):
        super().__init__(name, precedence)
        
        self.energy_init = jnp.array(energy_init)
        self.energy_max = jnp.array(energy_max)
        self.energy_burst = jnp.array(energy_burst)
        self.energy_decay = jnp.array(energy_decay)
        self.entity_type = entity_type
        self.subtype = subtype

    def get_step_function(self, state, neighbor_manager, key):

        entity_type = state.entity_type_to_int(self.entity_type)
        idxs = state.e_cond(self.entity_type)
        n_entities = state.entity_state.exists.shape[0]

        def state_fn(state, neighbors, key):
            entities = getattr(state, self.entity_type)

            mask = type_mask(state.entity_state, entity_type=entity_type, subtype=self.subtype)
            
            consumption_matrix = state.entity_state.consumption_matrix
            # Normalize mask by row sums, handling zero-sum rows
            row_sums = consumption_matrix.sum(axis=1, keepdims=True)
            mask_normalized = jnp.where(row_sums > 0, consumption_matrix / row_sums, 0)            

            # # `consuming` is the number of other entities consumed by each entity.
            # # It is a float, so that when multiple entities consume the same target,
            # # they only get the corresponding fraction of it.
            consuming = mask_normalized.sum(axis=1)
            
            neigh_flat = neighbors.idx.ravel()
            mask_normalized_flat = mask_normalized.ravel()
            
            # Similar logic as in consuming
            consumed = jax.ops.segment_sum(mask_normalized_flat, neigh_flat, n_entities)            
            
            
            energy = entities.energy + (consuming - consumed) * entities.energy_burst
            
            energy = jnp.where(
                mask,
                energy - entities.energy_decay,
                energy
            )

            # Redistribute the extra energy (positive or negative)
            # from entities outside bounds to those within bounds
            energy_outside_bounds = (energy <= 0) | (energy > entities.energy_max)
            bounds = jnp.zeros_like(energy)
            bounds = jnp.where(energy > entities.energy_max, entities.energy_max, bounds)            
            extra_energy = jnp.where(mask & energy_outside_bounds, energy - bounds, 0.).sum()
            per_entity = extra_energy / jnp.sum(mask & ~energy_outside_bounds)             
            energy = jnp.where(
                mask & ~energy_outside_bounds,
                energy + per_entity,
                energy
            )
            energy = jnp.where(
                energy_outside_bounds,
                jnp.clip(energy, 0, entities.energy_max),
                energy,
            )                                  
            
            return state.set(**{
                self.entity_type: entities.set(
                    energy=energy
                ),
                'entity_state': state.entity_state.set(
                    consumption_matrix=jnp.full(neighbor_manager.neighbors.idx.shape, False)
                    )
                }
            )

        return state_fn

    def update_state_cls(self, state_cls):
        superclass = state_cls.__annotations__[self.entity_type] if self.entity_type in state_cls.__annotations__ else object
        @md_dataclass
        class AgentState(superclass):
            energy: jnp.ndarray = None
            energy_init: jnp.ndarray = None
            energy_max: jnp.ndarray = None
            energy_burst: jnp.ndarray = None
            energy_decay: jnp.ndarray = None
        state_cls.__annotations__[self.entity_type] = AgentState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        if len(self.energy_init.shape) == 0:
            energy = jnp.full(getattr(state, self.entity_type).count(), self.energy_init)
        else:
            energy = self.energy_init
        return state.set(
            **{self.entity_type: getattr(state, self.entity_type).set(
                energy=energy,
                energy_init=self.energy_init,
                energy_max=self.energy_max,
                energy_burst=self.energy_burst,
                energy_decay=self.energy_decay,                
            )}
        )
        