import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.component import Component
from vivarium.environment.utils import type_mask


class EnergyComponent(Component):
    def __init__(self, name, precedence,
                 entity_type, subtype,
                 init_energy, max_energy, decay, burst):
        super().__init__(name, precedence)
        self.init_energy = init_energy
        self.max_energy = max_energy
        self.decay = decay
        self.burst = burst
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

            energy = jnp.where(
                jnp.logical_and(mask,
                                state.entity_state.consuming
                                ),
                cur_energy + self.burst,
                cur_energy
            )
            energy = jnp.where(
                mask,
                energy - self.decay,
                energy
            )

            energy = jnp.clip(energy, 0, self.max_energy)

            return state.set(**{
                self.entity_type: entities.set(
                    energy=energy[idxs]
                )}
            )

        return state_fn

    def update_state_cls(self, state_cls):
        assert 'entity_state' in state_cls.__annotations__, 'no entity_state in state class'
        # assert 'consuming' in state_cls.__annotations__['entity_state'].__annotations__, 'consuming not in entity_state'
        if self.entity_type in state_cls.__annotations__:
            @md_dataclass
            class AgentState(state_cls.__annotations__[self.entity_type]):
                energy: jnp.ndarray = None
        else:
            @md_dataclass
            class AgentState:
                energy: jnp.ndarray = None
        state_cls.__annotations__[self.entity_type] = AgentState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            **{self.entity_type: getattr(state, self.entity_type).set(
                energy=jnp.full(getattr(state, self.entity_type).count(), self.init_energy)
            )}
        )
        