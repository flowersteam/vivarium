import jax.numpy as jnp
from jax import vmap


@vmap
def normal(theta):
    """Returns the cos and the sin of an angle

    :param theta: angle in radians
    :return: cos and sin
    """
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def distance(displacement_fn, point1, point2):
    """Returns the distance between two points

    :param displacement_fn: displacement function (typically a jax_md.space function)
    :param point1: point 1
    :param point2: point 2
    :return: distance between the two points
    """
    diff = displacement_fn(point1, point2)
    squared_diff = jnp.sum(jnp.square(diff))
    return jnp.sqrt(squared_diff)


def relative_position(displ, theta):
    """
    Compute the relative distance and angle from a source particle to a target particle
    :param displ: Displacement vector (jnp arrray with shape (2,) from source to target
    :param theta: Orientation of the source particle (in the reference frame of the map)
    :return: dist: distance from source to target.
    relative_theta: relative angle of the target in the reference frame of the source particle (front direction at angle 0)
    """
    dist = jnp.linalg.norm(displ)
    norm_displ = displ / dist
    theta_displ = jnp.arccos(norm_displ[0]) * jnp.sign(jnp.arcsin(norm_displ[1]))
    relative_theta = theta_displ - theta
    return dist, relative_theta
