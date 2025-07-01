import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.entities.braitenberg.selective_sensing.dynamics import braitenberg_state_fn
from vivarium.environments.entities.component import EntityComponent
from vivarium.environments.environment import get_mask_fn
from vivarium.environments.state import BaseParticleState


@md_dataclass
class AgentState(BaseParticleState):
    prox: jnp.array
    prox_per_subtype: jnp.array
    motor: jnp.array
    behavior: jnp.array
    behavior_params: jnp.array
    sensed: jnp.array
    wheel_diameter: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array


class BraitenbergComponent(EntityComponent):

    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction,
                 exists, n_behaviors, n_subtypes,
                 wheel_diameter, proxs_dist_max, proxs_cos_min,
                 controller_kwargs=None):

        super().__init__(name=name, precedence=precedence,
                         entity_type=entity_type, subtype=subtype,
                         position=position, orientation=orientation,
                         mass=mass, diameter=diameter,
                         friction=friction, exists=exists)

        self.n_behaviors = n_behaviors
        self.n_subtypes = n_subtypes
        self.wheel_diameter = jnp.array(wheel_diameter)
        self.proxs_dist_max = jnp.array(proxs_dist_max)
        self.proxs_cos_min = jnp.array(proxs_cos_min)
        self.controller_kwargs = controller_kwargs

    @classmethod
    def get_kwargs(cls, name, scene_config, config_node, exclude=[]):
        kwargs = super().get_kwargs(name, scene_config, config_node, exclude)
        kwargs['n_subtypes'] = len(scene_config.config.subtypes)
        return kwargs

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
                           behavior=jnp.full((self.n_max, self.n_behaviors), 4, dtype=int),
                           behavior_params= jnp.zeros((self.n_max, self.n_behaviors, 2, 3)),
                           sensed=jnp.ones((self.n_max, self.n_behaviors, self.n_subtypes), dtype=int),
                           wheel_diameter=jnp.full((self.n_max,), self.wheel_diameter),
                           proxs_dist_max=jnp.full((self.n_max,), self.proxs_dist_max),
                           proxs_cos_min=jnp.full((self.n_max,), self.proxs_cos_min)
        )
        return state.set(
            **{self.entity_type: agent_state}
        )

    def get_step_function(self, state, neighbor_manager, key):
        braitenberg_mask = state.entity_state.entity_type == getattr(state, self.entity_type).entity_type
        return  braitenberg_state_fn(self.entity_type, braitenberg_mask, neighbor_manager.displacement, get_mask_fn('exists'))
