from enum import Enum

import jax.numpy as jnp


class Behaviors(Enum):
    FEAR = 0
    AGGRESSION = 1
    LOVE = 2
    SHY = 3
    NOOP = 4
    MANUAL = 5
    CUSTOM = 6

# Each row corresponds to a motor output (left or right)
# Each column corresponds to the linear weights [bias, left_prox, right_prox, left_motor, right_motor]
behavior_params = {
    Behaviors.FEAR: jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]),
    Behaviors.AGGRESSION: jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
    Behaviors.LOVE: jnp.array([[1.0, -1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0]]),
    Behaviors.SHY: jnp.array([[1.0, 0.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0]]),
    Behaviors.NOOP: jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
    Behaviors.MANUAL: jnp.array([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
    Behaviors.CUSTOM: jnp.array([[0.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0]]),
}
