import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import BaseParticleState
from vivarium.environments.environment import get_mask_fn
from vivarium.environments.entities.components import EntityComponent

from vivarium.environments.entities.braitenberg.selective_sensing.dynamics import braitenberg_state_fn

class BraitenbergComponent(EntityComponent):
    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction,
                 exists, n_behaviors, n_subtypes,
                 wheel_diameter, proxs_dist_max, proxs_cos_min):
        
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

    @classmethod
    def from_config(cls, name, scene_config, config_node):

        kwargs = cls.get_kwargs(name, scene_config, config_node)
        kwargs['n_subtypes'] = len(scene_config.config.subtypes)

        return cls(
            name=name,
            **kwargs
        )

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.entity_type] = AgentState
        setattr(state_cls, self.entity_type, None)
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        entity_idx = jnp.arange(self.offset, self.offset + self.n_max, dtype=int)
        agent_state = state.__annotations__[self.entity_type](
                           entity_type=self.entity_type_int,
                           entity_idx=entity_idx,
                           prox=jnp.zeros((self.n_max, 2)),
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
        ag_idx = state.entity_state.entity_type[neighbor_manager.neighbors.idx[0]] == self.entity_type_int
        agents_neighs_idx = neighbor_manager.neighbors.idx[:, ag_idx]

        # Give the idx of the agents in sparse representation, under a dense representation (used to get the raw proxs in compute motors function)
        agents_idx_dense_senders = jnp.array(
            [
                jnp.argwhere(jnp.equal(agents_neighs_idx[0, :], idx)).flatten()
                for idx in jnp.arange(getattr(state, self.entity_type).count())
            ]
        )
        # Note: jnp.argwhere(jnp.equal(self.agents_neighs_idx[0, :], idx)).flatten() ~ jnp.where(agents_idx[0, :] == idx)

        # Give the idx of the agent neighbors in dense representation
        agents_idx_dense_receivers = agents_neighs_idx[1, :][agents_idx_dense_senders]
        agents_idx_dense = agents_idx_dense_senders, agents_idx_dense_receivers
        return  braitenberg_state_fn(self.entity_type, neighbor_manager.displacement, get_mask_fn('exists'), 
                                        agents_neighs_idx, 
                                        agents_idx_dense, occlusion=True)


@md_dataclass
class AgentState(BaseParticleState):
    prox: jnp.array
    # prox_sensed_ent_type: jnp.array
    # prox_sensed_ent_idx: jnp.array
    motor: jnp.array
    behavior: jnp.array
    behavior_params: jnp.array
    sensed: jnp.array
    wheel_diameter: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array
