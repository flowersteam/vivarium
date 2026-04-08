import jax
from jax import lax
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.components.eco_evo.utils import spawn_entity
from vivarium.components.component import Component

_FLAT_KEYS = {'subtype', 'period', 'start', 'position_range', 'orientation_range'}


@md_dataclass
class SpawnState:
    subtype: jnp.ndarray          # shape [n_configs]
    period: jnp.ndarray           # shape [n_configs]
    start: jnp.ndarray            # shape [n_configs]
    position_range: jnp.ndarray   # shape [n_configs, 4]
    orientation_range: jnp.ndarray  # shape [n_configs, 2]


class SpawnComponent(Component):
    def __init__(self, name, precedence, **spawn_params):
        super().__init__(name, precedence)
        self.state_attr = f'{self.name}_state'

        # Backward compat: flat params (subtype, period, ...) -> single config "default"
        # Multi-spawn: detect if any value is a mapping (dict or OmegaConf DictConfig).
        # Use duck-typing so we don't need to import omegaconf here.
        is_mapping = lambda v: hasattr(v, 'keys')
        has_dict_config = any(is_mapping(v) for v in spawn_params.values())
        if has_dict_config:
            # Multi-spawn: keep only mapping-valued entries (ignore flat scalars from default.yaml)
            self.spawn_params_dict = {k: v for k, v in spawn_params.items() if is_mapping(v)}
        elif any(k in spawn_params for k in _FLAT_KEYS):
            # Legacy single-config: wrap as {"default": {...}}
            self.spawn_params_dict = {'default': spawn_params}
        else:
            self.spawn_params_dict = {}

    def init_state_fn(self, state, neighbor_manager, key):
        configs = list(self.spawn_params_dict.values())
        return state.set(
            **{self.state_attr: SpawnState(
                subtype=jnp.array([c['subtype'] for c in configs]),
                period=jnp.array([c['period'] for c in configs]),
                start=jnp.array([c['start'] for c in configs]),
                position_range=jnp.array([c['position_range'] for c in configs]),
                orientation_range=jnp.array([c['orientation_range'] for c in configs]),
            )}
        )

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.state_attr] = SpawnState
        setattr(state_cls, self.state_attr, None)
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        n_configs = len(self.spawn_params_dict)

        def state_fn(state, neighbors, key):
            spawn_state = getattr(state, self.state_attr)

            def apply_one_spawn(i, state):
                subtype_i = spawn_state.subtype[i]
                period_i = spawn_state.period[i]
                start_i = spawn_state.start[i]
                position_range_i = spawn_state.position_range[i]
                orientation_range_i = spawn_state.orientation_range[i]

                cond = jnp.logical_and(start_i, (state.time % period_i) == 0)
                key_i = jax.random.fold_in(key, i)

                return lax.cond(
                    cond,
                    lambda: spawn_entity(key_i, state,
                                        position_range_i,
                                        orientation_range_i,
                                        subtype=subtype_i),
                    lambda: state
                )

            return lax.fori_loop(0, n_configs, apply_one_spawn, state)

        return state_fn
