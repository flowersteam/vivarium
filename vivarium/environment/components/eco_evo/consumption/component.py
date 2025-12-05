import jax
import jax.numpy as jnp
from jax_md import partition
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.utils import get_relative_displacement
from vivarium.environment.components.component import Component
from vivarium.environment.utils import neighbors_entity_mask
from vivarium.environment.state import BaseEntityState


@md_dataclass
class ConsumptionState:
    source_subtype: jnp.ndarray
    target_subtype: jnp.ndarray
    range: jnp.ndarray
    start: jnp.ndarray


def count_masked_values(x, mask, num_values):
    """
    For each integer value in x, count how many times it corresponds 
    to True values in mask.
    
    Args:
        x: array of integers, shape (N,), with values in [0, num_values)
        mask: array of booleans, shape (N,)
        num_values: the range of possible values [0, num_values).
                   Must be a concrete integer (not a traced value).
    
    Returns:
        counts: array of shape (num_values,) where counts[i] is the number
                of times value i appears in x where mask is True
    """
    # Use segment_sum to count occurrences
    # Convert mask to integers (True -> 1, False -> 0)
    counts = jax.ops.segment_sum(
        mask.astype(jnp.int32),
        x,
        num_segments=num_values
    )
    
    return counts

count_masked_values = jax.jit(count_masked_values, static_argnums=(2,))

class ConsumptionComponent(Component):
    def __init__(self, name, precedence, source_subtype, target_subtype, range, start):
        super().__init__(name, precedence)
        self.source_subtype = source_subtype
        self.target_subtype = target_subtype
        self.range = range
        self.start = start
        self.state_attr = f'{self.name}_state'

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            entity_state=state.entity_state.set(
                consuming=jnp.full(state.entity_state.exists.shape, 0.),
                consumed=jnp.full(state.entity_state.exists.shape, 0.)
            ),
            **{self.state_attr: ConsumptionState(
                source_subtype=jnp.array(self.source_subtype),
                target_subtype=jnp.array(self.target_subtype),
                range=jnp.array(self.range),
                start=jnp.array(self.start)
            )}            
        )

    def update_state_cls(self, state_cls):
        base_cls = state_cls.__annotations__['entity_state'] if 'entity_state' in state_cls.__annotations__ else BaseEntityState
        @md_dataclass
        class EntityState(base_cls):
            consuming: jnp.ndarray = None
            consumed: jnp.ndarray = None
        state_cls.__annotations__['entity_state'] = EntityState
        state_cls.__annotations__[self.state_attr] = ConsumptionState
        setattr(state_cls, self.state_attr, None)        
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        source_mask = jnp.full(state.entity_state.exists.shape, True, dtype=bool)
        n_entities = state.entity_state.exists.shape[0]
        def step_fn(state, neighbors, key):
            
            consumption_state = getattr(state, self.state_attr)
            
            # TODO: use proximity map component instead?
            # d_r = state.distance_map
            d_r, _ = (
                get_relative_displacement(
                    state.entity_state.position,
                    state.entity_state.orientation,
                    source_mask,
                    neighbors.idx,
                    displacement_fn=neighbor_manager.displacement
                )
            )            

            mask = neighbors_entity_mask(
                neighbors_idx=neighbors.idx,
                source_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == consumption_state.source_subtype),
                target_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == consumption_state.target_subtype),
                neighbor_mask=partition.neighbor_list_mask(neighbors, mask_self=True)
            )
            mask &= jnp.logical_and(d_r < consumption_state.range, consumption_state.start)

            # Normalize mask by row sums, handling zero-sum rows
            row_sums = mask.sum(axis=1, keepdims=True)
            mask_normalized = jnp.where(row_sums > 0, mask / row_sums, 0)            

            # # `consuming` is the number of other entities consumed by each entity.
            # # It is a float, so that when multiple entities consume the same target,
            # # they only get the corresponding fraction of it.
            consuming = state.entity_state.consuming + mask_normalized.sum(axis=1)
            
            neigh_flat = neighbors.idx.ravel()
            mask_normalized_flat = mask_normalized.ravel()            
            
            # Similar logic as in consuming
            # TODO: if consuming entities are taking more than the energy available in a target, they should share only what is available
            # But this is rather related to the energy component than the consumption one. Maybe they should be merged.
            consumed = state.entity_state.consumed + jax.ops.segment_sum(mask_normalized_flat, neigh_flat, n_entities)

            return state.set(
                entity_state=state.entity_state.set(
                    # exists=new_exists,
                    consuming=consuming,
                    consumed=consumed
                )
            )

        return step_fn
