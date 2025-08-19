import logging as lg

import jax.numpy as jnp
from jax import vmap

from jax_md import partition

from vivarium.environments.components.entities.braitenberg.behaviors import Behaviors

from vivarium.environments.components.entities.braitenberg.simple.dynamics import (
    linear_behavior,
    motor_force,
    sum_force_to_entities
)
from vivarium.environments.utils import neighbors_entity_mask
from vivarium.environments.utils import get_relative_displacement


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


def compute_motor_selective(prox_per_subtype, sensed, params, behaviors, motors):
    prox = jnp.max(prox_per_subtype * sensed[jnp.newaxis, :], axis=1)
    return compute_motor(prox, params, behaviors, motors)


def left_or_right_prox(mask, dist, relative_theta, dist_max, cos_min, agent_neighbors):

    sensed = mask & (jnp.cos(relative_theta) > jnp.tile(cos_min, (relative_theta.shape[1], 1)).T)
    dist = jnp.where(sensed, dist, jnp.inf)
    min_dist_neigbor = jnp.argmin(dist, axis=1)
    distance_to_target = dist[jnp.arange(dist.shape[0]), min_dist_neigbor]
    prox = jnp.where(
        distance_to_target < dist_max,
        1. - distance_to_target / dist_max,
        0.
    )

    prox_idx = agent_neighbors[jnp.arange(agent_neighbors.shape[0]), jnp.argmin(dist, axis=1)]

    return prox, prox_idx


def compute_proxs(braitenberg_mask, source_mask, target_mask, neighbor_mask, neighbors_idx, displacement, positions, orientations, proxs_dist_max, proxs_cos_min):

    agent_neighbors = neighbors_idx[braitenberg_mask]
    mask = neighbors_entity_mask(agent_neighbors, source_mask, target_mask, neighbor_mask[braitenberg_mask])
    
    all_dist, all_relative_theta = (
        get_relative_displacement(
            positions,
            orientations[braitenberg_mask], 
            braitenberg_mask,
            agent_neighbors, 
            displacement_fn=displacement
        )
    )

    all_dist = jnp.where(
        mask, 
        all_dist, 
        jnp.inf)

    left_prox, left_idx = left_or_right_prox(
        mask & (jnp.sin(all_relative_theta) >= 0),
        all_dist,
        all_relative_theta,
        proxs_dist_max,
        proxs_cos_min,
        agent_neighbors
    )

    right_prox, right_idx = left_or_right_prox(
        mask & (jnp.sin(all_relative_theta) < 0),
        all_dist,
        all_relative_theta,
        proxs_dist_max,
        proxs_cos_min,
        agent_neighbors
    )

    return (
        jnp.vstack((left_prox, right_prox)).T,
        jnp.vstack((left_idx, right_idx)).T
    )


def extend_prox_per_subtype(prox, prox_idx, entity_subtype, n_subtypes):
    subtype_mask = entity_subtype[prox_idx][:, :, jnp.newaxis] == jnp.arange(n_subtypes)[jnp.newaxis, jnp.newaxis, :]
    return prox[:, :, jnp.newaxis] * subtype_mask


def braitenberg_state_fn(braitenberg_state_field, braitenberg_mask, displacement, mask_fn):

    def state_fn(state, neighbors, key):

        exists_mask = mask_fn(state)

        braitenberg_state = getattr(state, braitenberg_state_field)

        neighbor_mask = partition.neighbor_list_mask(neighbors, mask_self=True)

        proxs, prox_idx = compute_proxs(
            braitenberg_mask=braitenberg_mask,
            source_mask=state.entity_state.exists[braitenberg_state.entity_idx],
            target_mask=state.entity_state.exists,
            neighbor_mask=neighbor_mask,
            neighbors_idx=neighbors.idx,
            displacement=displacement,
            positions=state.entity_state.position,
            orientations=state.entity_state.orientation,
            proxs_dist_max=braitenberg_state.proxs_dist_max,
            proxs_cos_min=braitenberg_state.proxs_cos_min
        )

        prox_per_subtype = extend_prox_per_subtype(
            proxs, prox_idx, state.entity_state.entity_subtype, braitenberg_state.sensed.shape[-1]
        )

        motors = vmap(vmap(compute_motor_selective, (None, 0, 0, 0, None)))(prox_per_subtype, braitenberg_state.sensed, braitenberg_state.behavior_params, braitenberg_state.behavior, braitenberg_state.motor)

        motors = jnp.mean(motors, axis=1)

        # # Update agents state
        braitenberg_state = braitenberg_state.set(
            prox=proxs,
            prox_per_subtype=prox_per_subtype,
            # proximity_map_dist=proximity_dist_map,
            # proximity_map_theta=proximity_dist_theta,
            motor=motors,
        )

        # # Update the entities and the state
        state = state.set(**{braitenberg_state_field: braitenberg_state})

        center, orientation = motor_force(state, braitenberg_state, exists_mask)

        return state.set(entity_state=sum_force_to_entities(state.entity_state, center, orientation))
    
    return state_fn
