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

        def state_fn(state, neighbors, key):
            entities = getattr(state, self.entity_type)

            cur_energy = jnp.zeros(state.entity_state.exists.shape)
            cur_energy = cur_energy.at[idxs].set(entities.energy)

            mask = type_mask(state.entity_state, entity_type=entity_type, subtype=self.subtype)
            
            
            energy = cur_energy + (state.entity_state.consuming - state.entity_state.consumed) * entities.energy_burst
            
            
            energy = jnp.where(
                mask,
                energy - entities.energy_decay,
                energy
            )

            energy = jnp.clip(energy, 0, entities.energy_max)

            return state.set(**{
                self.entity_type: entities.set(
                    energy=energy[idxs]                    
                ),
                'entity_state': state.entity_state.set(
                    consuming=jnp.full(state.entity_state.exists.shape, 0.),
                    consumed=jnp.full(state.entity_state.exists.shape, 0.)
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
        