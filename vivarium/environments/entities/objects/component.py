import jax.numpy as jnp

from vivarium.environments.entities.component import EntityComponent
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environments.state import BaseParticleState


from jax_md.dataclasses import dataclass as md_dataclass


@md_dataclass
class ObjectState(BaseParticleState):
    pass


class ControllerObject(ControllerEntity):
    pass
# TODO: What's the purpose of this class? Not uses at the moment (May, 31, 2025) but the whole pipeline seems to work anyway.
# (Idem in braintenberg component.py)
class PanelControllerObject(ControllerObject):
    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)


class ObjectComponent(EntityComponent):

    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction,
                 exists,
                 controller_kwargs=None):

        super().__init__(name=name, precedence=precedence,
                         entity_type=entity_type, subtype=subtype,
                         position=position, orientation=orientation,
                         mass=mass, diameter=diameter,
                         friction=friction, exists=exists)

        self.controller_kwargs = controller_kwargs

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.entity_type] = ObjectState
        setattr(state_cls, self.entity_type, None)
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        self.entity_idx = jnp.arange(self.offset, self.offset + self.n_max, dtype=int)
        object_state = state.__annotations__[self.entity_type](
                           entity_type=self.entity_type_int,
                           entity_idx=self.entity_idx
        )
        return state.set(
            **{self.entity_type: object_state}
        )
