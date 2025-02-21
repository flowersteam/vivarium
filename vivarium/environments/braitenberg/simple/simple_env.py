import jax.numpy as jnp

from jax import vmap
from jax import random, ops

from jax_md import space, rigid_body

from vivarium.environments.base_env import BaseEnv, NeighborManager
from vivarium.environments.utils import normal, relative_position

from vivarium.environments.physics_engine import (
    reset_force_state_fn,
    collision_state_fn,
    friction_state_fn,
    init_state_fn,
    step_state_fn
)
from vivarium.environments.braitenberg.simple import init_state
from vivarium.environments.braitenberg.behaviors import Behaviors
from vivarium.environments.braitenberg.simple.state import EntityType


### Define the constants and the classes of the environment to store its state ###
SPACE_NDIMS = 2


# --- 1 Functions to compute the proximeter of braitenberg agents ---#
proximity_map = vmap(relative_position, (0, 0))


def sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists):
    """
    Compute the proximeter activations (left, right) induced by the presence of an entity
    :param dist: distance from the agent to the entity
    :param relative_theta: angle of the entity in the reference frame of the agent (front direction at angle 0)
    :param dist_max: Max distance of the proximiter (will return 0. above this distance)
    :param cos_min: Field of view as a cosinus (e.g. cos_min = 0 means a pi/4 FoV on each proximeter, so pi/2 in total)
    :return: left and right proximeter activation in a jnp array with shape (2,)
    """
    cos_dir = jnp.cos(relative_theta)
    prox = 1.0 - (dist / dist_max)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, jnp.sin(relative_theta) >= 0)
    left = in_view * at_left * prox
    right = in_view * (1.0 - at_left) * prox
    return jnp.array([left, right]) * target_exists  # i.e. 0 if target does not exist


sensor_fn = vmap(sensor_fn, (0, 0, 0, 0, 0))


def sensor(dist, relative_theta, dist_max, cos_min, max_agents, senders, target_exists):
    """Return the sensor values of all agents

    :param dist: relative distances between agents and targets
    :param relative_theta: relative angles between agents and targets
    :param dist_max: maximum range of proximeters
    :param cos_min: cosinus of proximeters angles
    :param max_agents: number of agents
    :param senders: indexes of agents sensing the environment
    :param target_exists: mask to indicate which sensed entities exist or not
    :return: proximeter activations
    """
    raw_proxs = sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists)
    # Computes the maximum within the proximeter activations of agents on all their neigbhors.
    proxs = ops.segment_max(raw_proxs, senders, max_agents)

    return proxs


def compute_prox(state, agents_neighs_idx, target_exists_mask, displacement):
    """
    Set agents' proximeter activations
    :param state: full simulation State
    :param agents_neighs_idx: Neighbor representation, where sources are only agents. Matrix of shape (2, n_pairs),
    where n_pairs is the number of neighbor entity pairs where sources (first row) are agent indexes.
    :param target_exists_mask: Specify which target entities exist. Vector with shape (n_entities,).
    target_exists_mask[i] is True (resp. False) if entity of index i in state.entities exists (resp. don't exist).
    :return:
    """
    center = state.entities.position_center
    orientation = state.entities.position_orientation
    mask = target_exists_mask[agents_neighs_idx[1, :]]
    senders, receivers = agents_neighs_idx
    Ra = center[senders]
    Rb = center[receivers]
    dR = -space.map_bond(displacement)(
        Ra, Rb
    )  # Looks like it should be opposite, but don't understand why

    # Create distance and angle maps between entities
    dist, theta = proximity_map(dR, orientation[senders])
    proximity_map_dist = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_dist = proximity_map_dist.at[senders, receivers].set(dist)
    proximity_map_theta = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_theta = proximity_map_theta.at[senders, receivers].set(theta)

    prox = sensor(
        dist,
        theta,
        state.agents.proxs_dist_max[senders],
        state.agents.proxs_cos_min[senders],
        len(state.agents.ent_idx),
        senders,
        mask,
    )

    return prox, proximity_map_dist, proximity_map_theta


def linear_behavior(proxs, params):
    """Compute the activation of motors with a linear combination of proximeters and parameters

    :param proxs: proximeter values of an agent
    :param params: parameters of an agent (mapping proxs to motor values)
    :return: motor values
    """
    return params.dot(jnp.hstack((proxs, 1.0)))


v_linear_behavior = vmap(linear_behavior, in_axes=(0, 0))


def compute_motor(proxs, params, behaviors, motors):
    """Compute new motor values. If behavior is manual, keep same motor values. Else, compute new values with proximeters and params.

    :param proxs: proximeters of all agents
    :param params: parameters mapping proximeters to new motor values
    :param behaviors: array of behaviors
    :param motors: current motor values
    :return: new motor values
    """
    manual = jnp.where(behaviors == Behaviors.MANUAL.value, 1, 0)
    manual_mask = jnp.broadcast_to(jnp.expand_dims(manual, axis=1), motors.shape)
    linear_motor_values = v_linear_behavior(proxs, params)
    motor_values = linear_motor_values * (1 - manual_mask) + motors * manual_mask
    return motor_values


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


def motor_force(state, mask):
    """Returns the motor force function of the environment

    :param state: state
    :param mask: mask on entities (e.g. existing ones)
    :return: motor force
    """
    agent_idx = state.agents.ent_idx

    n = normal(state.entities.unified_orientation[agent_idx])

    fwd, rot = motor_command(state.agents.motor,
                             state.entities.diameter[agent_idx],
                             state.agents.wheel_diameter)

    cur_vel = (
        state.entities.unified_momentum[agent_idx]
        / state.entities.unified_mass[agent_idx]
    )

    cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)

    fwd_delta = fwd - cur_fwd_vel

    fwd_force = (
        n
        * jnp.tile(fwd_delta, (SPACE_NDIMS, 1)).T
        * jnp.tile(state.agents.speed_mul, (SPACE_NDIMS, 1)).T
    )

    center = (
        jnp.zeros_like(state.entities.unified_position).at[agent_idx].set(fwd_force)
    )

    # TODO CMF: if I get rid of RigidBody, do I also get rid of mass.orientation?
    if state.entities.is_rigid_body():
        cur_rot_vel = (
            state.entities.momentum.orientation[agent_idx]
            / state.entities.mass.orientation[agent_idx]
        )
        rot_delta = rot - cur_rot_vel
        rot_force = rot_delta * state.agents.theta_mul
    else:
        rot_force = state.dt * rot

    orientation = (
        jnp.zeros_like(state.entities.unified_orientation)
        .at[agent_idx]
        .set(rot_force)
    )

    orientation = jnp.where(mask, orientation, 0.0)
    mask = jnp.stack([mask] * SPACE_NDIMS, axis=1)
    center = jnp.where(mask, center, 0.0)

    return center, orientation


def sum_force_to_entities(entities, center, orientation=0.):
    if not entities.is_rigid_body():
        return entities.set(force=center + entities.force, orientation=orientation + entities.orientation)
    else:
        center += entities.force.center
        orientation += entities.force.orientation         
        return entities.set(force=rigid_body.RigidBody(center=center, orientation=orientation))
        


def braitenberg_state_fn(displacement, mask_fn, agents_neighs_idx):
    def state_fn(state, neighbors):
        exists_mask = mask_fn(state)
        prox, proximity_dist_map, proximity_dist_theta = compute_prox(
            state,
            agents_neighs_idx,
            target_exists_mask=exists_mask,
            displacement=displacement,
        )

        motor = compute_motor(
            prox, state.agents.params, state.agents.behavior, state.agents.motor
        )
        agents = state.agents.set(
            prox=prox,
            proximity_map_dist=proximity_dist_map,
            proximity_map_theta=proximity_dist_theta,
            motor=motor,
        )

        state = state.set(agents=agents)

        center, orientation = motor_force(state, exists_mask)

        return state.set(entities=sum_force_to_entities(state.entities, center, orientation))
    return state_fn


class BraitenbergEnv(BaseEnv):
    def __init__(self, state, space_fn=space.periodic, occlusion=True, seed=42):
        
        displacement, shift = space_fn(state.box_size)

        exists_mask_fn = lambda state: state.entities.exists == 1
        key = random.PRNGKey(seed)
        key, new_key = random.split(key)
        init_fn = init_state_fn(key)
        neighbor_manager = NeighborManager(displacement, state)
        ag_idx = state.entities.entity_type[neighbor_manager.neighbors.idx[0]] == EntityType.AGENT.value
        agents_neighs_idx = neighbor_manager.neighbors.idx[:, ag_idx]
        state_fns = [reset_force_state_fn(),
                     braitenberg_state_fn(displacement, exists_mask_fn, 
                                          agents_neighs_idx),
                     collision_state_fn(displacement, exists_mask_fn),
                     friction_state_fn(exists_mask_fn),
                     step_state_fn(shift, exists_mask_fn, new_key)]
        super().__init__(state, init_fn, state_fns, neighbor_manager)


if __name__ == "__main__":
    state = init_state()
    env = BraitenbergEnv(state)
    for _ in range(10):
        state = env.step(state)
