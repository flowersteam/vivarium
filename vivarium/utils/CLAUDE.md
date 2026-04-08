# vivarium/utils/

## Purpose

Utility package providing runtime abstraction, process management, configuration loading, dataclass proxying, and update checking. Serves all other packages — no vivarium dependencies outside this package.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Re-exports public API from `handle_server_interface` |
| `runtime.py` | Solid | Frozen vs dev mode detection, path resolution, command builders |
| `handle_server_interface.py` | Brittle | Server/interface/Jupyter/ngrok process lifecycle |
| `scene_configs.py` | Solid | Hydra config loading, scene enumeration, position generation |
| `dataclass_wrapper.py` | Solid (complex) | Remote proxy for collecting state changes; JAX-aware dataclass updates |
| `updater.py` | Solid (complex) | PyInstaller update checking, download, post-update merge |
| `converters.py` | Solid | Class path import, CamelCase/snake_case conversion |
| `timer.py` | Solid | Frequency-respecting sleep timer |
| `jax_utils.py` | Solid | JAX-MD dataclass detection |

## Public API

### From `__init__.py`
`stop_server_and_interface`, `start_simulation_server`, `stop_simulation_server`, `start_panel_interface`, `stop_panel_interface`, `kill_vivarium_processes`, `kill_all_vivarium_processes`, `wait_for_http`, `check_server_running`

### Commonly imported from submodules
- **`handle_server_interface`**: `start_simulation_server`, `stop_simulation_server`, `start_panel_interface`, `stop_panel_interface`, `start_jupyter_server`, `stop_jupyter_server`, `wait_for_grpc_server`, `check_server_running`, `kill_vivarium_processes`, `kill_all_vivarium_processes`, `create_ngrok_tunnel`, `close_ngrok_tunnel`, `check_colab_environment`
- **`scene_configs`**: `load_scene_config`, `load_config`, `get_available_scenes`, `component_factories_from_config`, `compute_parameters`
- **`runtime`**: `get_config_dir`, `get_app_root`, `is_frozen`, `initialize_user_data`, `get_version`
- **`dataclass_wrapper`**: `Remote`, `update_dataclass`, `create_dataclass_from_dict`
- **`timer`**: `SleepTimer`, `sleep_timer`
- **`updater`**: `check_for_updates`, manifest/merge functions
- **`converters`**: `import_class`, `upper_camel_to_snake`, `snake_to_upper_camel`

## Internal Dependency Graph

```
runtime.py (foundation — no utils deps)
  ↑ updater.py, scene_configs.py, handle_server_interface.py

jax_utils.py (standalone)
  ↑ dataclass_wrapper.py

converters.py, timer.py (standalone)
```

## Who Imports From Utils

- **Scripts**: `run_server.py`, `run_interface.py`
- **Simulator**: `simulator.py`, `simulator_client.py`
- **Controllers**: `vivarium_controller.py`, `controller.py`, `utils.py`
- **Environment**: `environment.py`, `components/entities/component.py`
- **Interface**: `panel_app.py`
- **Tests**: Multiple test files
- **Notebooks**: Tutorials and sessions

## Test Coverage

| Module | Test file | What's tested | Gaps |
|--------|-----------|---------------|------|
| `runtime.py` | `test_runtime.py`, `test_version.py` | Path resolution, frozen mode paths, `initialize_user_data`, version file parsing, version comparison | Command builders (`get_server_command`, etc.) untested |
| `updater.py` | `test_update_check.py` | `check_for_updates`, `get_defaults_update_info`, `find_latest_backup_dir`, `perform_post_update_merge` | `download_update`, `apply_downloaded_update`, disk space check untested (hard to unit test) |
| `dataclass_wrapper.py` | `test_dataclass_api.py`, `test_grpc.py` | `Remote` proxy, dataclass updates, simulator integration | Nested/recursive edge cases not explicitly tested |
| `scene_configs.py` | `test_scene_config.py` | Config loading, component factories, environment/simulator creation | `get_available_scenes`, position/orientation generators, `compute_parameters` untested |
| `handle_server_interface.py` | `test_start_stop_scripts.py` | Server start/stop, server+interface start/stop | Jupyter lifecycle, ngrok, port killing, process PID lookup — all untested |
| `converters.py` | — | — | No dedicated tests (exercised indirectly by environment tests) |
| `timer.py` | — | — | No dedicated tests |
| `jax_utils.py` | — | — | No dedicated tests (trivial, exercised indirectly) |

**Summary**: Core modules (`runtime`, `updater`, `dataclass_wrapper`) have good coverage. `handle_server_interface` is barely tested — only the happy path of start/stop. `scene_configs` has partial coverage. Three small modules have no dedicated tests but are exercised indirectly.

