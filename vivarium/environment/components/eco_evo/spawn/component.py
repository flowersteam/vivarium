from jax import lax
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.eco_evo.utils import spawn_entity
from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import f32

@md_dataclass
class SpawnState:
    subtype: jnp.ndarray
    period: jnp.ndarray
    position_range: jnp.ndarray
    orientation_range: jnp.ndarray

class SpawnComponent(Component):
    def __init__(self, name, precedence, subtype, period, position_range, orientation_range):
        super().__init__(name, precedence)
        self.subtype = subtype
        self.period = period
        self.position_range = position_range
        self.orientation_range = orientation_range
        self.state_attr = f'{self.name}_state'

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            **{self.state_attr: SpawnState(
                subtype=jnp.array(self.subtype),
                period=jnp.array(self.period),
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

            cond = (state.time % getattr(state, self.state_attr).period) == 0

            return lax.cond(
                cond,
                lambda: spawn_entity(key, state, 
                                     getattr(state, self.state_attr).position_range, 
                                     getattr(state, self.state_attr).orientation_range, 
                                     subtype=getattr(state, self.state_attr).subtype),
                lambda: state
            )

        return state_fn