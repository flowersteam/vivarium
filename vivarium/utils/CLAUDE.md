# vivarium/utils/ — Package Audit

_Audited: 2026-03-10 | Status: Phase 1 Step 1_

## Purpose

Utility package providing runtime abstraction, process management, configuration loading, dataclass proxying, and update checking. Serves all other packages — no vivarium dependencies outside this package.

## Files

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `__init__.py` | 10 | Solid | Re-exports public API from `handle_server_interface` |
| `runtime.py` | 242 | Solid | Frozen vs dev mode detection, path resolution, command builders |
| `handle_server_interface.py` | 881 | Brittle | Server/interface/Jupyter/ngrok process lifecycle |
| `scene_configs.py` | 201 | Solid | Hydra config loading, scene enumeration, position generation |
| `dataclass_wrapper.py` | 174 | Solid (complex) | Remote proxy for collecting state changes; JAX-aware dataclass updates |
| `updater.py` | 783 | Solid (complex) | PyInstaller update checking, download, post-update merge |
| `converters.py` | 54 | Solid | Class path import, CamelCase/snake_case conversion |
| `timer.py` | 53 | Solid | Frequency-respecting sleep timer |
| `jax_utils.py` | 11 | Solid | JAX-MD dataclass detection |

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

- **Scripts**: `run_server.py`, `run_interface.py`, `run_vivarium.py`
- **Simulator**: `simulator.py`, `simulator_client.py`
- **Controllers**: `vivarium_controller.py`, `controller.py`, `utils.py`
- **Environment**: `environment.py`, `components/entities/component.py`
- **Interface**: `panel_app.py`
- **Tests**: Multiple test files
- **Notebooks**: Tutorials and sessions

## Known Issues

### High Severity

1. **`handle_server_interface.py` is a monolith (881 lines)**
   Mixes process management, server lifecycle, Jupyter support, and ngrok. Hard to maintain and test. Could be split into `_process.py`, `_server.py`, `_interface.py`, `_jupyter.py`, `_ngrok.py` behind a re-exporting `__init__.py`.

2. **Missing `start_server_and_interface()` function**
   Some notebooks import it but it doesn't exist. Only `start_simulation_server()` + `start_panel_interface()` exist separately.

### Medium Severity

3. **Scene enumeration is hardcoded** in `get_available_scenes()` — uses string matching (`startswith('session')`, etc.). New scene types require code changes.

4. **`dataclass_wrapper.py` has complex recursive logic** in `Remote._rec()` — hard to debug but works.

5. **No checksum validation for downloaded updates** in `updater.py`.

### Low Severity

6. **`get_process_pids_unix/windows` are public** but should be internal (`_`-prefixed).

7. **Jupyter port 8889 hardcoded** in multiple places — no named constant.

8. **Some helper functions lack docstrings** (`_get_ssl_context`, `_version_is_newer`, `_build_defaults_manifest`).

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

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Split `handle_server_interface.py` into submodules |
| High | Add `start_server_and_interface()` convenience function, or consider removing it |
| Medium | Make platform-specific PID functions internal |
| Medium | Move scene category patterns to config dict |
| Low | Add type hints to `handle_server_interface.py` |
| Low | Extract ngrok to optional submodule |
