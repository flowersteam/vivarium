import jax.numpy as jnp

from jax import vmap

from jax_md import rigid_body

from vivarium.environments.utils import normal

#TODO: merge simple and selective_sensing packages (now we only use selective_sensing)

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
        