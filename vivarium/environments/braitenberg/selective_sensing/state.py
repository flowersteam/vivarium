import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import ParticleState


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


@md_dataclass
class ObjectState(ParticleState):
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type)
