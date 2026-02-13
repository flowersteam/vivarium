from jax import vmap
import jax.numpy as jnp
from jax_md import partition
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.utils import get_relative_displacement
from vivarium.environment.components.component import Component
from vivarium.environment.utils import neighbors_entity_mask


@md_dataclass
class ConsumptionState:
    source_subtype: jnp.ndarray
    target_subtype: jnp.ndarray
    range: jnp.ndarray
    start: jnp.ndarray
    consumption_matrix: jnp.ndarray = None


def single_consumption(d_r, neighbors_idx, neighbor_mask, exists, entity_subtype, diameter, source_subtype, target_subtype, start, range):     

    mask = neighbors_entity_mask(
        neighbors_idx=neighbors_idx,
        source_mask=jnp.logical_and(exists == 1, entity_subtype == source_subtype),
        target_mask=jnp.logical_and(exists == 1, entity_subtype == target_subtype),
        neighbor_mask=neighbor_mask
    )
    source_target_radius_sum = (jnp.tile(diameter[:, jnp.newaxis], (1, neighbors_idx.shape[1])) + diameter[neighbors_idx]) / 2
    mask &= jnp.logical_and(d_r - source_target_radius_sum < range, start)
    
    # Normalize mask by row sums, handling zero-sum rows
    row_sums = mask.sum(axis=1, keepdims=True)
    mask_normalized = jnp.where(row_sums > 0, mask / row_sums, 0)               

    return mask_normalized


single_consumption = vmap(single_consumption, in_axes=(None, None, None, None, None, None, 0, 0, 0, 0))


class ConsumptionComponent(Component):
    def __init__(self, name, precedence, subtype_labels=None, consuming_in_entity_state=False, **consumption_params):
        super().__init__(name, precedence)
        self.consumption_params_dict = consumption_params
        self.state_attr = f'{self.name}_state'
        self.consuming_in_entity_state = consuming_in_entity_state
        self.subtype_labels = subtype_labels

    def update_state_cls(self, state_cls):
        if self.consuming_in_entity_state:
            @md_dataclass
            class EntityState(state_cls.__annotations__['entity_state']):
                consuming: jnp.ndarray = None
            state_cls.__annotations__['entity_state'] = EntityState
            setattr(state_cls, 'entity_state', None)
        state_cls.__annotations__[self.state_attr] = ConsumptionState
        setattr(state_cls, self.state_attr, None)
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        for params in self.consumption_params_dict.values():
            for k in ['source_subtype', 'target_subtype']:
                if isinstance(params[k], str):
                    assert self.subtype_labels is not None, "subtype_labels must be provided to use string subtype names"
                    params[k] = self.subtype_labels.index(params[k])
        state = state.set(
            **{self.state_attr: ConsumptionState(
                source_subtype=jnp.array([params['source_subtype'] for params in self.consumption_params_dict.values()]),
                target_subtype=jnp.array([params['target_subtype'] for params in self.consumption_params_dict.values()]),
                range=jnp.array([params['range'] for params in self.consumption_params_dict.values()]),
                start=jnp.array([params['start'] for params in self.consumption_params_dict.values()]),
                consumption_matrix=jnp.full(neighbor_manager.neighbors.idx.shape, 0.)
            )}            
        )

        if self.consuming_in_entity_state:
            state = state.set(
                entity_state=state.entity_state.set(
                    consuming=jnp.full(state.entity_state.exists.shape[0], False, dtype=bool),
                )
            )
        return state

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        source_mask = jnp.full(state.entity_state.exists.shape, True, dtype=bool)
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
            
            consumption_matrix = single_consumption(
                d_r,
                neighbors.idx,
                partition.neighbor_list_mask(neighbors, mask_self=True),
                state.entity_state.exists,
                state.entity_state.entity_subtype,
                state.entity_state.diameter,
                consumption_state.source_subtype,
                consumption_state.target_subtype,
                consumption_state.start,
                consumption_state.range
            ).sum(axis=0)                    

            state = state.set(
                **{self.state_attr: getattr(state, self.state_attr).set(
                    consumption_matrix=consumption_matrix
                )}
            )
            
            if self.consuming_in_entity_state:
                consuming = jnp.any(consumption_matrix > 0, axis=1)
                state = state.set(
                    entity_state=state.entity_state.set(
                        consuming=consuming,
                    )
                )
            
            return state

        return step_fn
    
    def neighbor_update(self, state, neighbor_manager, key):
        return state.set(
            **{self.state_attr: getattr(state, self.state_attr).set(
                consumption_matrix=jnp.full(neighbor_manager.neighbors.idx.shape, 0.)
            )}
        )
