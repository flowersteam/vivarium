import logging as lg

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax_md.rigid_body import RigidBody
from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate


@md_dataclass
class BaseState:
    time: jnp.int32
    box_size: jnp.int32


@md_dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    exists: jnp.array

    def is_rigid_body(self):
        return hasattr(self.position, 'center')

    def __getattr__(self, name):
        if name.startswith("unified_"):
            attr = name[len("unified_"):]
            if attr == 'orientation':
                if isinstance(self.position, RigidBody):
                    return self.position.orientation
                return self.orientation
            if isinstance(getattr(self, attr), RigidBody):
                return getattr(self, attr).center
            return getattr(self, attr)           
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

@md_dataclass
class BaseParticleState:
    ent_idx: jnp.array

@md_dataclass
class Neighbors:
    neighbors: jnp.array
    agents_neighs_idx: jnp.array
    agents_idx_dense: jnp.array


class BaseEnv:
    def __init__(self):
        raise (NotImplementedError)

    def init_state(self) -> BaseState:
        raise (NotImplementedError)

    @partial(jit, static_argnums=(0,))
    def _step_env(
        self, state: BaseState, neighbors_storage: Neighbors
    ) -> Tuple[BaseState, Neighbors]:
        raise (NotImplementedError)

    def step(self, state: BaseState, num_updates) -> BaseState:
        raise (NotImplementedError)

    def allocate_neighbors(self, state: BaseState, position=None):
        position = state.entities.unified_position if position is None else position
        neighbors = self.neighbor_fn.allocate(position)
        return neighbors
