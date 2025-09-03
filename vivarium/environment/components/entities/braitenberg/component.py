import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.entities.braitenberg.sensorimotor import braitenberg_state_fn
from vivarium.environment.components.entities.component import EntityComponent
from vivarium.environment.environment import MaskFunction
from vivarium.environment.state import BaseParticleState


@md_dataclass
class AgentState(BaseParticleState):
    prox: jnp.array
    prox_per_subtype: jnp.array
    motor: jnp.array
    behavior_params: jnp.array
    sensed_mask: jnp.array
    wheel_diameter: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array


class BraitenbergComponent(EntityComponent):

    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction,
                 exists, n_behaviors, n_subtypes,
                 wheel_diameter, proxs_dist_max, proxs_cos_min,
                 subtype_labels=None, controller_kwargs=None):

        super().__init__(name=name, precedence=precedence,
                         entity_type=entity_type, subtype=subtype,
                         position=position, orientation=orientation,
                         mass=mass, diameter=diameter,
                         friction=friction, exists=exists, 
                         subtype_labels=subtype_labels)

        self.n_behaviors = n_behaviors
        self.n_subtypes = n_subtypes
        self.wheel_diameter = jnp.array(wheel_diameter)
        self.proxs_dist_max = jnp.array(proxs_dist_max)
        self.proxs_cos_min = jnp.array(proxs_cos_min)
        self.controller_kwargs = controller_kwargs

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'n_behaviors': self.n_behaviors,
            'n_subtypes': self.n_subtypes,  # Do we need this? (in the yaml config it computes it from ${scene.subtypes})
            'wheel_diameter': getattr(state, self.entity_type).wheel_diameter.tolist(),
            'proxs_dist_max': getattr(state, self.entity_type).proxs_dist_max.tolist(),
            'proxs_cos_min': getattr(state, self.entity_type).proxs_cos_min.tolist(),
        })
        return config

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.entity_type] = AgentState
        setattr(state_cls, self.entity_type, None)
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        self.entity_idx = jnp.arange(self.offset, self.offset + self.n_max, dtype=int)
        agent_state = state.__annotations__[self.entity_type](
                           entity_type=self.entity_type_int,
                           entity_idx=self.entity_idx,
                           prox=jnp.zeros((self.n_max, 2)),
                           prox_per_subtype=jnp.zeros((self.n_max, 2, self.n_subtypes)),
                           motor=jnp.zeros((self.n_max, 2)),
                           behavior_params= jnp.zeros((self.n_max, self.n_behaviors, 2, 3)),
                           sensed_mask=jnp.ones((self.n_max, self.n_behaviors, self.n_subtypes), dtype=int),
                           wheel_diameter=jnp.full((self.n_max,), self.wheel_diameter),
                           proxs_dist_max=jnp.full((self.n_max,), self.proxs_dist_max),
                           proxs_cos_min=jnp.full((self.n_max,), self.proxs_cos_min)
        )
        return state.set(
            **{self.entity_type: agent_state}
        )

    def get_step_function(self, state, neighbor_manager, key):
        braitenberg_mask = state.entity_state.entity_type == getattr(state, self.entity_type).entity_type
        return  braitenberg_state_fn(self.entity_type, braitenberg_mask, neighbor_manager.displacement, MaskFunction('exists'))
