import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import ParticleState

from vivarium.environments.environment import exists_mask_fn

from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import braitenberg_state_fn

def get_state_function(state, neighbor_manager):
    braitenberg_attr_name = state.field_name(AgentState)
    assert braitenberg_attr_name is not None, "No braitenberg agent found in state"
    ag_idx = state.entity_state.entity_type[neighbor_manager.neighbors.idx[0]] == state.entity_type_to_int(braitenberg_attr_name)
    agents_neighs_idx = neighbor_manager.neighbors.idx[:, ag_idx]

    # Give the idx of the agents in sparse representation, under a dense representation (used to get the raw proxs in compute motors function)
    agents_idx_dense_senders = jnp.array(
        [
            jnp.argwhere(jnp.equal(agents_neighs_idx[0, :], idx)).flatten()
            for idx in jnp.arange(getattr(state, braitenberg_attr_name).count())
        ]
    )
    # Note: jnp.argwhere(jnp.equal(self.agents_neighs_idx[0, :], idx)).flatten() ~ jnp.where(agents_idx[0, :] == idx)

    # Give the idx of the agent neighbors in dense representation
    agents_idx_dense_receivers = agents_neighs_idx[1, :][agents_idx_dense_senders]
    agents_idx_dense = agents_idx_dense_senders, agents_idx_dense_receivers

    return braitenberg_state_fn(braitenberg_attr_name, neighbor_manager.displacement, exists_mask_fn, 
                                agents_neighs_idx, 
                                agents_idx_dense, occlusion=True)

@md_dataclass
class AgentState(ParticleState):
    prox: jnp.array
    prox_sensed_ent_type: jnp.array
    prox_sensed_ent_idx: jnp.array
    motor: jnp.array
    proximity_map_dist: jnp.array
    proximity_map_theta: jnp.array
    behavior: jnp.array
    behavior_params: jnp.array
    sensed: jnp.array
    wheel_diameter: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array
    
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        
        agent_kwargs = entity_types_kwargs[entity_type]
        n_entities = agent_kwargs['n_exists']
        n_total_entities = sum([kwargs['n_exists'] for kwargs in entity_types_kwargs.values()])
        n_behaviors = agent_kwargs['n_behaviors']
        n_subtypes = 0
        for etype, kwargs in entity_types_kwargs.items():
            for st, n in kwargs['subtype_to_n']:
                n_subtypes = max(n_subtypes, st + 1)
        proximity_map_dist = jnp.zeros((n_entities, n_total_entities))
        proximity_map_theta = jnp.zeros((n_entities, n_total_entities))
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type,
                           prox=jnp.zeros((n_entities, 2)),
                           motor=jnp.zeros((n_entities, 2)),
                           prox_sensed_ent_type=jnp.zeros((n_entities, 2), dtype=int),
                           prox_sensed_ent_idx=jnp.zeros((n_entities, 2), dtype=int),
                           proximity_map_dist=proximity_map_dist,
                           proximity_map_theta=proximity_map_theta,
                           behavior=jnp.full((n_entities, n_behaviors), 4, dtype=int),
                           behavior_params= jnp.zeros((n_entities, n_behaviors, 2, 3)),
                           sensed=jnp.ones((n_entities, n_behaviors, n_subtypes), dtype=int) #TODO: Check if this is correct
                           )
    def state_fns(self):
        return [get_state_function]

@md_dataclass
class ObjectState(ParticleState):
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type)
