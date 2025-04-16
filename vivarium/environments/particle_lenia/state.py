import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import ParticleState


@md_dataclass
class ParticleLeniaState(ParticleState):
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type)
