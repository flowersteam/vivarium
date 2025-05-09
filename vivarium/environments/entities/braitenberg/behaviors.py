from enum import Enum

import jax.numpy as jnp


class Behaviors(Enum):
    FEAR = 0
    AGGRESSION = 1
    LOVE = 2
    SHY = 3
    NOOP = 4
    MANUAL = 5


behavior_params = {
    Behaviors.FEAR.value: jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    Behaviors.AGGRESSION.value: jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    Behaviors.LOVE.value: jnp.array([[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]]),
    Behaviors.SHY.value: jnp.array([[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]]),
    Behaviors.NOOP.value: jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    Behaviors.MANUAL.value: jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}


def behavior_to_params(behavior):
    """Return the params associated to a behavior.

    :param behavior: behavior id (int)
    :return: params
    """
    return behavior_params[behavior]
