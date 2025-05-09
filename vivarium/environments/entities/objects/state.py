from vivarium.environments.state import ParticleState


from jax_md.dataclasses import dataclass as md_dataclass


@md_dataclass
class ObjectState(ParticleState):
    @classmethod
    def create(cls, entity_idx_offset, entity_types_kwargs, entity_type):
        return cls._create(entity_idx_offset, entity_types_kwargs, entity_type)