from collections.abc import Iterable

import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environments.state import ParticleState


@md_dataclass
class ObjectState(ParticleState):
    mu_k: jnp.array
    sigma_k: jnp.array
    w_k: jnp.array
    mu_g: jnp.array
    sigma_g: jnp.array
    c_rep: jnp.array
    @classmethod
    def create(cls, entity_idx_offset, params, entity_type_field):
        for attr, val in params['state_data'][entity_type_field]['kwargs'].items():
            if attr in [field.name for field in cls.__dataclass_fields__.values()]:
                if not isinstance(val, Iterable) or len(val) == 1:
                    params['state_data'][entity_type_field]['kwargs'][attr] = [val] * params['state_data'][entity_type_field]['kwargs']['n_exists']
        return cls._create(entity_idx_offset, params, entity_type_field)
