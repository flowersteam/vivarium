from jax import lax
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.eco_evo.utils import spawn_entity
from vivarium.environment.components.component import Component

@md_dataclass
class SpawnState:
    subtype: jnp.ndarray
    period: jnp.ndarray
    start: jnp.ndarray
    position_range: jnp.ndarray
    orientation_range: jnp.ndarray

class SpawnComponent(Component):
    def __init__(self, name, precedence, subtype, period, start, position_range, orientation_range):
        super().__init__(name, precedence)
        self.subtype = subtype
        self.period = period
        self.start = start
        self.position_range = position_range
        self.orientation_range = orientation_range
        self.state_attr = f'{self.name}_state'

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            **{self.state_attr: SpawnState(
                subtype=jnp.array(self.subtype),
                period=jnp.array(self.period),
                start=jnp.array(self.start),
                position_range=jnp.array(self.position_range),
                orientation_range=jnp.array(self.orientation_range)
            )}
        )

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.state_attr] = SpawnState
        setattr(state_cls, self.state_attr, None)
        return state_cls


    def get_step_function(self, state, neighbor_manager, key):

        def state_fn(state, neighbors, key):
            
            spawn_state = getattr(state, self.state_attr)

            cond = jnp.logical_and(spawn_state.start, (state.time % spawn_state.period) == 0)

            return lax.cond(
                cond,
                lambda: spawn_entity(key, state, 
                                     spawn_state.position_range, 
                                     spawn_state.orientation_range, 
                                     subtype=spawn_state.subtype),
                lambda: state
            )

        return state_fn