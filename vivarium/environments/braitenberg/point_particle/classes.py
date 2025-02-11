from enum import Enum

import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.base_env import BaseState, BaseEntityState

from vivarium.environments.braitenberg.selective_sensing.classes import (
    EntityType,
    EntityState as RigidbodyEntityState,
    ParticleState,
    AgentState,
    ObjectState,
    State,
    Neighbors
)



@md_dataclass
class EntityState(RigidbodyEntityState):
    orientation: jnp.array



