import logging as lg

import jax
import jax.numpy as jnp

from jax import vmap, jit
from jax import random, lax
from jax_md import space

from vivarium.environments.base_env import BaseEnv, NeighborManager
from vivarium.environments.physics_engine import (
    reset_force_state_fn,
    collision_state_fn,
    friction_state_fn,
    init_state_fn,
    step_state_fn
)
from vivarium.environments.braitenberg.behaviors import Behaviors
from vivarium.environments.braitenberg.selective_sensing import (
    State,
    EntityType,
    init_state
)

from vivarium.environments.braitenberg.simple.simple_env import (
    proximity_map,
    sensor_fn,
    linear_behavior,
    motor_force,
    sum_force_to_entities
)


SPACE_NDIMS = 2


# TODO : Should refactor the function to split the returns
def get_relative_displacement(state, agents_neighs_idx, displacement_fn):
    """Get all infos relative to distance and orientation between all agents and their neighbors

    :param state: state
    :param agents_neighs_idx: idx all agents neighbors
    :param displacement_fn: jax md function enabling to know the distance between points
    :return: distance array, angles array, distance map for all agents, angles map for all agents
    """
    # body = state.entities.position
    position = state.entities.unified_position
    orientation = state.entities.unified_orientation
    senders, receivers = agents_neighs_idx
    Ra = position[senders]
    Rb = position[receivers]
    dR = -space.map_bond(displacement_fn)(
        Ra, Rb
    )  # Looks like it should be opposite, but don't understand why

    dist, theta = proximity_map(dR, orientation[senders])
    proximity_map_dist = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_dist = proximity_map_dist.at[senders, receivers].set(dist)
    proximity_map_theta = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_theta = proximity_map_theta.at[senders, receivers].set(theta)
    return dist, theta, proximity_map_dist, proximity_map_theta


def compute_motor(proxs, params, behaviors, motors):
    """Compute new motor values. If behavior is manual, keep same motor values. Else, compute new values with proximeters and params.

    :param proxs: proximeters of all agents
    :param params: parameters mapping proximeters to new motor values
    :param behaviors: array of behaviors
    :param motors: current motor values
    :return: new motor values
    """
    manual = jnp.where(behaviors == Behaviors.MANUAL.value, 1, 0)
    manual_mask = manual
    linear_motor_values = linear_behavior(proxs, params)
    motor_values = linear_motor_values * (1 - manual_mask) + motors * manual_mask
    return motor_values


### 1 : Functions for selective sensing with occlusion


def update_mask(mask, left_n_right_types, ent_type):
    """Update a mask of

    :param mask: mask that will be applied on sensors of agents
    :param left_n_right_types: types of left adn right sensed entities
    :param ent_type: entity subtype (e.g 1 for predators)
    :return: mask
    """
    cur = jnp.where(left_n_right_types == ent_type, 0, 1)
    mask *= cur
    return mask


def keep_mask(mask, left_n_right_types, ent_type):
    """Return the mask unchanged

    :param mask: mask
    :param left_n_right_types: left_n_right_types
    :param ent_type: ent_type
    :return: mask
    """
    return mask


def mask_proxs_occlusion(proxs, left_n_right_types, ent_sensed_arr):
    """Mask the proximeters of agents with occlusion

    :param proxs: proxiemters of agents without occlusion (shape = (2,))
    :param e_sensed_types: types of both entities sensed at left and right (shape=(2,))
    :param ent_sensed_arr: mask of sensed subtypes by the agent (e.g jnp.array([0, 1, 0, 1]) if sense only entities of subtype 1 and 4)
    :return: updated proximeters according to sensed_subtypes
    """
    mask = jnp.array([1, 1])
    # Iterate on the array of sensed entities mask
    for ent_type, sensed in enumerate(ent_sensed_arr):
        # If an entity is sensed, update the mask, else keep it as it is
        mask = jax.lax.cond(
            sensed, update_mask, keep_mask, mask, left_n_right_types, ent_type
        )
    # Update the mask with 0s where the mask is, else keep the prox value
    proxs = jnp.where(mask, 0, proxs)
    return proxs


# Example :
# ent_sensed_arr = jnp.array([0, 1, 0, 0, 1])
# proxs = jnp.array([0.8, 0.2])
# e_sensed_types = jnp.array([4, 4]) # Modify these values to check it works
# print(mask_proxs_occlusion(proxs, e_sensed_types, ent_sensed_arr))


def compute_behavior_motors(
    state, params, sensed_mask, behavior, motor, agent_proxs, sensed_ent_idx
):
    """Compute the motor values for a specific behavior

    :param state: state
    :param params: behavior params params
    :param sensed_mask: sensed_mask for this behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_proxs: agent proximeters (unmasked)
    :param sensed_ent_idx: idx of left and right entities sensed
    :return: right motor values for this behavior
    """
    left_n_right_types = state.entities.ent_subtype[sensed_ent_idx]
    behavior_proxs = mask_proxs_occlusion(agent_proxs, left_n_right_types, sensed_mask)
    motors = compute_motor(behavior_proxs, params, behaviors=behavior, motors=motor)
    return motors


# See for the vectorizing idx because already in a vmaped function here
compute_all_behavior_motors = vmap(
    compute_behavior_motors, in_axes=(None, 0, 0, 0, None, None, None)
)


def compute_occlusion_proxs_motors(
    state,
    agent_idx,
    params,
    sensed,
    behaviors,
    motor,
    raw_proxs,
    ag_idx_dense_senders,
    ag_idx_dense_receivers,
):
    """_summary_

    :param state: state
    :param agent_idx: agent idx in entities
    :param params: params arrays for all agent's behaviors
    :param sensed: sensed mask arrays for all agent's behaviors
    :param behaviors: agent behaviors array
    :param motor: agent motors
    :param raw_proxs: raw_proximeters for all agents (shape=(n_agents * (n_entities - 1), 2))
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: _description_
    """
    behavior = jnp.expand_dims(behaviors, axis=1)
    # Compute the neighbors idx of the agent and get its raw proximeters (of shape (n_entities -1 , 2))
    ent_ag_neighs_idx = ag_idx_dense_senders[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_neighs_idx]

    # Get the max and arg max of these proximeters on axis 0, gives results of shape (2,)
    agent_proxs = jnp.max(agent_raw_proxs, axis=0)
    argmax = jnp.argmax(agent_raw_proxs, axis=0)
    # Get the real entity idx of the left and right sensed entities from dense neighborhoods
    sensed_ent_idx = ag_idx_dense_receivers[agent_idx][argmax]
    prox_sensed_ent_types = state.entities.ent_subtype[sensed_ent_idx]

    # Compute the motor values for all behaviors and do a mean on it
    motor_values = compute_all_behavior_motors(
        state, params, sensed, behavior, motor, agent_proxs, sensed_ent_idx
    )
    motors = jnp.mean(motor_values, axis=0)

    return agent_proxs, (sensed_ent_idx, prox_sensed_ent_types), motors


compute_all_agents_proxs_motors_occl = vmap(
    compute_occlusion_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None)
)


### 2 : Functions for selective sensing without occlusion


def mask_sensors(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Mask the raw proximeters of agents for a specific entity type

    :param state: state
    :param agent_raw_proxs: raw_proximeters of agent (shape=(n_entities - 1), 2)
    :param ent_type_id: entity subtype id (e.g 0 for PREYS)
    :param ent_neighbors_idx: idx of agent neighbors in entities arrays
    :return: updated agent raw proximeters
    """
    mask = jnp.where(state.entities.ent_subtype[ent_neighbors_idx] == ent_type_id, 0, 1)
    mask = jnp.expand_dims(mask, 1)
    mask = jnp.broadcast_to(mask, agent_raw_proxs.shape)
    return agent_raw_proxs * mask


def dont_change(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Leave the agent raw_proximeters unchanged

    :param state: state
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_type_id: ent_type_id
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: agent_raw_proxs
    """
    return agent_raw_proxs


def compute_behavior_prox(state, agent_raw_proxs, ent_neighbors_idx, sensed_entities):
    """Compute the proximeters for a specific behavior

    :param state: state
    :param agent_raw_proxs: agent raw proximeters
    :param ent_neighbors_idx: idx of agent neighbors
    :param sensed_entities: array of sensed entities
    :return: updated proximeters
    """
    # iterate over all the types in sensed_entities and return if they are sensed or not
    for ent_type_id, sensed in enumerate(sensed_entities):
        # change the proxs if you don't perceive the entity, else leave them unchanged
        agent_raw_proxs = lax.cond(
            sensed,
            dont_change,
            mask_sensors,
            state,
            agent_raw_proxs,
            ent_type_id,
            ent_neighbors_idx,
        )
    # Compute the final proxs with a max on the updated raw_proxs
    proxs = jnp.max(agent_raw_proxs, axis=0)
    return proxs


def compute_behavior_proxs_motors(
    state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx
):
    """Return the proximeters and the motors for a specific behavior

    :param state: state
    :param params: params of the behavior
    :param sensed: sensed mask of the behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: behavior proximeters, behavior motors
    """
    behavior_prox = compute_behavior_prox(
        state, agent_raw_proxs, ent_neighbors_idx, sensed
    )
    behavior_motors = compute_motor(behavior_prox, params, behavior, motor)
    return behavior_prox, behavior_motors


# vmap on params, sensed and behavior (parallelize on all agents behaviors at once, but not motorrs because are the same)
compute_all_behavior_proxs_motors = vmap(
    compute_behavior_proxs_motors, in_axes=(None, 0, 0, 0, None, None, None)
)


def compute_agent_proxs_motors(
    state,
    agent_idx,
    params,
    sensed,
    behavior,
    motor,
    raw_proxs,
    ag_idx_dense_senders,
    ag_idx_dense_receivers,
):
    """Compute the agent proximeters and motors for all behaviors

    :param state: state
    :param agent_idx: idx of the agent in entities
    :param params: array of params for all behaviors
    :param sensed: array of sensed mask for all behaviors
    :param behavior: array of behaviors
    :param motor: motor values
    :param raw_proxs: raw_proximeters of all agents
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: array of agent_proximeters, mean of behavior motors
    """
    behavior = jnp.expand_dims(behavior, axis=1)
    ent_ag_idx = ag_idx_dense_senders[agent_idx]
    ent_neighbors_idx = ag_idx_dense_receivers[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_idx]

    # vmap on params, sensed, behaviors and motorss (vmap on all agents)
    agent_proxs, agent_motors = compute_all_behavior_proxs_motors(
        state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx
    )
    mean_agent_motors = jnp.mean(agent_motors, axis=0)

    # need to return a dummy array as 2nd argument to match the compute_agent_proxs_motors function returns with occlusion
    dummy = (jnp.zeros(1), jnp.zeros(1))
    return agent_proxs, dummy, mean_agent_motors


compute_all_agents_proxs_motors = vmap(
    compute_agent_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None)
)

def braitenberg_state_fn(displacement, mask_fn, agents_neighs_idx, agents_idx_dense, occlusion=True):
    
    assert occlusion, "Non occlusion not working yet"
    prox_motor_function = compute_all_agents_proxs_motors_occl if occlusion else compute_all_agents_proxs_motors
     
    def state_fn(state, neighbors):

        # Retrieve different neighbors format
        senders, receivers = agents_neighs_idx
        ag_idx_dense_senders, ag_idx_dense_receivers = agents_idx_dense

        exists_mask = mask_fn(state)

        # Compute raw proxs for all agents first
        dist, relative_theta, proximity_dist_map, proximity_dist_theta = (
            get_relative_displacement(
                state, agents_neighs_idx, displacement_fn=displacement
            )
        )

        dist_max = state.agents.proxs_dist_max[senders]
        cos_min = state.agents.proxs_cos_min[senders]
        # changed agents_neighs_idx[1, :] to receivers in line below (check if it works)
        target_exist_mask = state.entities.exists[receivers]
        # Compute agents raw proximeters (proximeters for all neighbors)
        raw_proxs = sensor_fn(
            dist, relative_theta, dist_max, cos_min, target_exist_mask
        )

        # Compute real agents proximeters and motors
        agent_proxs, prox_sensed_ent_tuple, mean_agent_motors = (
            prox_motor_function(
                state,
                state.agents.ent_idx,
                state.agents.params,
                state.agents.sensed,
                state.agents.behavior,
                state.agents.motor,
                raw_proxs,
                ag_idx_dense_senders,
                ag_idx_dense_receivers,
            )
        )

        prox_sensed_ent_idx, prox_sensed_ent_type = prox_sensed_ent_tuple

        # Update agents state
        agents = state.agents.set(
            prox=agent_proxs,
            prox_sensed_ent_type=prox_sensed_ent_type,
            prox_sensed_ent_idx=prox_sensed_ent_idx,
            proximity_map_dist=proximity_dist_map,
            proximity_map_theta=proximity_dist_theta,
            motor=mean_agent_motors,
        )

        # Update the entities and the state
        state = state.set(agents=agents)

        center, orientation = motor_force(state, exists_mask)

        return state.set(entities=sum_force_to_entities(state.entities, center, orientation))
    
    return state_fn
    

# TODO : Fix the non occlusion error in the step
class SelectiveSensorsEnv(BaseEnv):
    def __init__(self, state, space_fn=space.periodic, occlusion=True, seed=42):
        
        displacement, shift = space_fn(state.box_size)

        exists_mask_fn = lambda state: state.entities.exists == 1
        key = random.PRNGKey(seed)
        key, new_key = random.split(key)
        init_fn = init_state_fn(key)
        neighbor_manager = NeighborManager(displacement, state)
        ag_idx = state.entities.entity_type[neighbor_manager.neighbors.idx[0]] == EntityType.AGENT.value
        agents_neighs_idx = neighbor_manager.neighbors.idx[:, ag_idx]

        # Give the idx of the agents in sparse representation, under a dense representation (used to get the raw proxs in compute motors function)
        agents_idx_dense_senders = jnp.array(
            [
                jnp.argwhere(jnp.equal(agents_neighs_idx[0, :], idx)).flatten()
                for idx in jnp.arange(state.max_agents)
            ]
        )
        # Note: jnp.argwhere(jnp.equal(self.agents_neighs_idx[0, :], idx)).flatten() ~ jnp.where(agents_idx[0, :] == idx)

        # Give the idx of the agent neighbors in dense representation
        agents_idx_dense_receivers = agents_neighs_idx[1, :][agents_idx_dense_senders]
        agents_idx_dense = agents_idx_dense_senders, agents_idx_dense_receivers

        state_fns = [reset_force_state_fn(),
                     braitenberg_state_fn(displacement, exists_mask_fn, 
                                          agents_neighs_idx, 
                                          agents_idx_dense, occlusion=occlusion),
                     collision_state_fn(displacement, exists_mask_fn),
                     friction_state_fn(exists_mask_fn),
                     step_state_fn(shift, exists_mask_fn, new_key)]
        super().__init__(state, init_fn, state_fns, neighbor_manager)


if __name__ == "__main__":
    state = init_state()
    env = SelectiveSensorsEnv(state)

    env.step(state, num_scan_steps=5)
    env.step(state, num_scan_steps=6)
