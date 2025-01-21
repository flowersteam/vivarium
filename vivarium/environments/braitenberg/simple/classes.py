from enum import Enum

import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate

from vivarium.environments.base_env import BaseState


class EntityType(Enum):
    AGENT = 0
    OBJECT = 1


# Already incorporates position, momentum, force, mass and velocity
@md_dataclass
class EntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    diameter: jnp.array
    friction: jnp.array
    exists: jnp.array


@md_dataclass
class ParticleState:
    ent_idx: jnp.array
    color: jnp.array


@md_dataclass
class AgentState(ParticleState):
    prox: jnp.array
    motor: jnp.array
    proximity_map_dist: jnp.array
    proximity_map_theta: jnp.array
    behavior: jnp.array
    params: jnp.array
    wheel_diameter: jnp.array
    speed_mul: jnp.array
    max_speed: jnp.array
    theta_mul: jnp.array
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array


@md_dataclass
class ObjectState(ParticleState):
    pass


@md_dataclass
class State(BaseState):
    max_agents: jnp.int32
    max_objects: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    collision_alpha: jnp.float32
    collision_eps: jnp.float32
    entities: EntityState
    agents: AgentState
    objects: ObjectState
