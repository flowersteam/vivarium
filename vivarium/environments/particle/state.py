import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass
from vivarium.environments.base_env import BaseEntityState
from vivarium.environments.base_env import BaseState



@md_dataclass
class ParticleState(BaseEntityState):
    idx: jnp.array
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    exists: jnp.array
    friction: jnp.array
    diameter: jnp.array

@md_dataclass
class State(BaseState):
    max_particles: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    collision_alpha: jnp.float32
    collision_eps: jnp.float32
    entities: ParticleState
