# vivarium/utils/

## Purpose

Utility package providing configuration loading, dataclass proxying, timers, and JAX helpers. Serves all other packages — no vivarium dependencies outside this package.

Process management, path resolution, and update utilities have moved to `vivarium/runtime/`.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Package marker (no re-exports) |
| `scene_configs.py` | Solid | Hydra config loading, scene enumeration, position generation |
| `dataclass_wrapper.py` | Solid (complex) | Remote proxy for collecting state changes; JAX-aware dataclass updates |
| `timer.py` | Solid | Frequency-respecting sleep timer |
| `jax_utils.py` | Solid | JAX-MD dataclass detection |

## Public API

### Commonly imported from submodules
- **`scene_configs`**: `load_scene_config`, `load_config`, `get_available_scenes`, `component_factories_from_config`, `compute_parameters`
- **`dataclass_wrapper`**: `Remote`, `update_dataclass`, `create_dataclass_from_dict`
- **`timer`**: `SleepTimer`, `sleep_timer`

## Internal Dependency Graph

```
jax_utils.py (standalone)
  ↑ dataclass_wrapper.py

timer.py (standalone)
scene_configs.py ← vivarium.runtime.paths.get_config_dir
```

## Who Imports From Utils

- **Scripts**: `run_server.py`
- **Simulator**: `simulator.py`, `simulator_client.py`, `grpc_server/converters.py`
- **Controllers**: `vivarium_controller.py`, `controller.py`
- **Environment**: `environment.py`, `components/entities/component.py`
- **Interface**: `panel_app.py`
- **Tests**: Multiple test files
- **Notebooks**: Tutorials and sessions

## Test Coverage

| Module | Test file | What's tested | Gaps |
|--------|-----------|---------------|------|
| `dataclass_wrapper.py` | `test_dataclass_api.py`, `test_grpc.py` | `Remote` proxy, dataclass updates, simulator integration | Nested/recursive edge cases not explicitly tested |
| `scene_configs.py` | `test_scene_config.py` | Config loading, component factories, environment/simulator creation | `get_available_scenes`, position/orientation generators, `compute_parameters` untested |
| `timer.py` | — | — | No dedicated tests |
| `jax_utils.py` | — | — | No dedicated tests (trivial, exercised indirectly) |

**Summary**: `dataclass_wrapper` has good coverage. `scene_configs` has partial coverage. Two small modules have no dedicated tests but are exercised indirectly.
