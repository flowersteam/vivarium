import jax.numpy as jnp
from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate
from vivarium.environments.base_env import BaseState



@md_dataclass
class ParticleState(simulate.NVEState):
    idx: jnp.array
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    exists: jnp.array

@md_dataclass
class State(BaseState):
    max_particles: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    particle_state: ParticleState

    @property
    def entities(self):
        return self.particle_state