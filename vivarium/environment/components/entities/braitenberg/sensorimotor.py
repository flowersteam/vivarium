import logging as lg

import jax.numpy as jnp
from jax import vmap

from jax_md import partition, rigid_body

from vivarium.environment.utils import normal
from vivarium.environment.utils import neighbors_entity_mask
from vivarium.environment.utils import get_relative_displacement
from vivarium.environment.components.entities.braitenberg.behaviors import Behaviors


### Define the constants and the classes of the environment to store its state ###
SPACE_NDIMS = 2


def linear_behavior(proxs, params):
    """Compute the activation of motors with a linear combination of proximeters and parameters

    :param proxs: proximeter values of an agent
    :param params: parameters of an agent (mapping proxs to motor values)
    :return: motor values
    """
    return params.dot(jnp.hstack((proxs, 1.0)))


v_linear_behavior = vmap(linear_behavior, in_axes=(0, 0))


def lr_2_fwd_rot(left_spd, right_spd, base_length, wheel_diameter):
    """Return the forward and angular speeds according the the speeds of left and right wheels

    :param left_spd: left wheel speed
    :param right_spd: right wheel speed
    :param base_length: distance between two wheels (diameter of the agent)
    :param wheel_diameter: diameter of wheels
    :return: forward and angular speeds
    """
    fwd = (wheel_diameter / 4.0) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_length) * (right_spd - left_spd)
    return fwd, rot


def fwd_rot_2_lr(fwd, rot, base_length, wheel_diameter):
    """Return the left and right wheels speeds according to the forward and angular speeds

    :param fwd: forward speed
    :param rot: angular speed
    :param base_length: distance between wheels (diameter of agent)
    :param wheel_diameter: diameter of wheels
    :return: left wheel speed, right wheel speed
    """
    left = ((2.0 * fwd) - (rot * base_length)) / wheel_diameter
    right = ((2.0 * fwd) + (rot * base_length)) / wheel_diameter
    return left, right


def motor_command(wheel_activation, base_length, wheel_diameter):
    """Return the forward and angular speed according to wheels speeds

    :param wheel_activation: wheels speeds
    :param base_length: distance between wheels
    :param wheel_diameter: wheel diameters
    :return: forward and angular speeds
    """
    fwd, rot = lr_2_fwd_rot(
        wheel_activation[0], wheel_activation[1], base_length, wheel_diameter
    )
    return fwd, rot


motor_command = vmap(motor_command, (0, 0, 0))


def motor_force(state, braitenberg_state, mask):
    """Returns the motor force function of the environment

    :param state: state
    :param braitenberg_state: braitenberg state (usually state.agents)
    :param mask: mask on entities (e.g. existing ones)
    :return: motor force
    """
    agent_idx = braitenberg_state.entity_idx

    n = normal(state.entity_state.unified_orientation[agent_idx])

    fwd, rot = motor_command(braitenberg_state.motor,
                             state.entity_state.diameter[agent_idx],
                             braitenberg_state.wheel_diameter)

    cur_vel = (
        state.entity_state.unified_momentum[agent_idx]
        / state.entity_state.unified_mass[agent_idx]
    )

    cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)

    fwd_delta = fwd - cur_fwd_vel

    fwd_force = (
        n
        * jnp.tile(fwd_delta, (SPACE_NDIMS, 1)).T
    )

    center = (
        jnp.zeros_like(state.entity_state.unified_position).at[agent_idx].set(fwd_force)
    )

    # TODO CMF: if I get rid of RigidBody, do I also get rid of mass.orientation?
    if state.entity_state.is_rigid_body():
        cur_rot_vel = (
            state.entity_state.momentum.orientation[agent_idx]
            / state.entity_state.mass.orientation[agent_idx]
        )
        rot_delta = rot - cur_rot_vel
        rot_force = rot_delta  # * state.agent_state.theta_mul
    else:
        rot_force = state.dt * rot

    orientation = (
        jnp.zeros_like(state.entity_state.unified_orientation)
        .at[agent_idx]
        .set(rot_force)
    )

    orientation = jnp.where(mask, orientation, 0.0)
    mask = jnp.stack([mask] * SPACE_NDIMS, axis=1)
    center = jnp.where(mask, center, 0.0)

    return center, orientation


def sum_force_to_entities(entity_state, center, orientation=0.):
    if not entity_state.is_rigid_body():
        return entity_state.set(force=center + entity_state.force, orientation=orientation + entity_state.orientation)
    else:
        center += entity_state.force.center
        orientation += entity_state.force.orientation         
        return entity_state.set(force=rigid_body.RigidBody(center=center, orientation=orientation))
        

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
