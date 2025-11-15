from vivarium.environment.components.component import Component
from vivarium.environment.state import BaseEntityState
from vivarium.environment.utils import neighbors_entity_mask


import jax.numpy as jnp
from jax_md import partition
from jax_md.dataclasses import dataclass as md_dataclass


class ConsumptionComponent(Component):
    def __init__(self, name, precedence, source_subtype, target_subtype, range):
        super().__init__(name, precedence)
        self.source_subtype = source_subtype
        self.target_subtype = target_subtype
        self.range = range

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            entity_state=state.entity_state.set(
                consuming=jnp.full(state.entity_state.exists.shape, False),
                consumed=jnp.full(state.entity_state.exists.shape, False)
            )
        )

    def update_state_cls(self, state_cls):
        base_cls = state_cls.__annotations__['entity_state'] if 'entity_state' in state_cls.__annotations__ else BaseEntityState
        @md_dataclass
        class EntityState(base_cls):
            consuming: jnp.ndarray = None
            consumed: jnp.ndarray = None
        state_cls.__annotations__['entity_state'] = EntityState
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        def step_fn(state, neighbors, key):
            d_r = state.distance_map

            mask = neighbors_entity_mask(
                neighbors_idx=neighbors.idx,
                source_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == self.source_subtype),
                target_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == self.target_subtype),
                neighbor_mask=partition.neighbor_list_mask(neighbors, mask_self=True)
            )
            mask &= d_r < self.range

            consuming = mask.any(axis=1)
            consumed = jnp.full(state.entity_state.exists.shape, False)

            neigh_flat = neighbors.idx.ravel()
            mask_flat = mask.ravel()
            consumed = consumed.at[neigh_flat].max(mask_flat)

            new_exists = jnp.where(
                consumed,
                0,
                state.entity_state.exists
            )

            return state.set(
                entity_state=state.entity_state.set(
                    exists=new_exists,
                    consuming=consuming,
                    consumed=consumed
                )
            )

        return step_fn