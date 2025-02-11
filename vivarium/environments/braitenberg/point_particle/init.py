from vivarium.environments.braitenberg.selective_sensing.init import *

from vivarium.environments.braitenberg.point_particle.classes import (
    AgentState,
    EntityState,
)

rigid_body_init_entities = init_entities
rigid_body_init_agents = init_agents
rigid_body_init_objects = init_objects
rigid_body_init_state = init_state


def init_entities_from_rigid_body(rigid_body_entity_state):
    return EntityState(
        position=rigid_body_entity_state.position.center,
        orientation=rigid_body_entity_state.position.orientation,
        momentum=rigid_body_entity_state.momentum,
        force=rigid_body_entity_state.force.center,
        mass=rigid_body_entity_state.mass.center,
        entity_type=rigid_body_entity_state.entity_type,
        ent_subtype=rigid_body_entity_state.ent_subtype,
        entity_idx=rigid_body_entity_state.entity_idx,
        diameter=rigid_body_entity_state.diameter,
        friction=rigid_body_entity_state.friction,
        exists=rigid_body_entity_state.exists,
    )

def init_entities(*args, **kwargs):
    rigid_body_entity_state = rigid_body_init_entities(*args, **kwargs)

    return init_entities_from_rigid_body(rigid_body_entity_state)


def init_agents_from_rigid_body(rigid_body_agent_state):
    return AgentState(
        ent_idx=rigid_body_agent_state.ent_idx,
        prox=rigid_body_agent_state.prox,
        prox_sensed_ent_type=rigid_body_agent_state.prox_sensed_ent_type,
        prox_sensed_ent_idx=rigid_body_agent_state.prox_sensed_ent_idx,
        motor=rigid_body_agent_state.motor,
        behavior=rigid_body_agent_state.behavior,
        params=rigid_body_agent_state.params,
        sensed=rigid_body_agent_state.sensed,
        wheel_diameter=rigid_body_agent_state.wheel_diameter,
        speed_mul=rigid_body_agent_state.speed_mul,
        max_speed=rigid_body_agent_state.max_speed,
        theta_mul=rigid_body_agent_state.theta_mul,
        proxs_dist_max=rigid_body_agent_state.proxs_dist_max,
        proxs_cos_min=rigid_body_agent_state.proxs_cos_min,
        proximity_map_dist=rigid_body_agent_state.proximity_map_dist,
        proximity_map_theta=rigid_body_agent_state.proximity_map_theta,
        color=rigid_body_agent_state.color,
    )

def init_agents(*args, **kwargs):
    rigid_body_agent_state = rigid_body_init_agents(*args, **kwargs)
    return init_agents_from_rigid_body(rigid_body_agent_state)


def init_state_from_rigid_body(rigid_body_state):
    entities = init_entities_from_rigid_body(rigid_body_state.entities)
    agents = init_agents_from_rigid_body(rigid_body_state.agents)
    objects = init_objects(rigid_body_state.max_agents, rigid_body_state.max_objects, rigid_body_state.objects.color)

    return init_complete_state(entities, agents, objects, 
                               rigid_body_state.max_agents, rigid_body_state.max_objects, 
                               rigid_body_state.ent_sub_types, 
                               rigid_body_state.box_size, 
                               rigid_body_state.neighbor_radius, 
                               rigid_body_state.collision_alpha, rigid_body_state.collision_eps, 
                               rigid_body_state.dt)

def init_state(*args, **kwargs):
    rigid_body_state = rigid_body_init_state(*args, **kwargs)

    return init_state_from_rigid_body(rigid_body_state)
