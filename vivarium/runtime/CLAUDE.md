# vivarium/runtime/

## Purpose

Runtime package for deployment and process lifecycle management. Provides path resolution for frozen (PyInstaller) vs development modes, subprocess management for the gRPC server, Panel interface, and Jupyter, ngrok tunneling for Google Colab, and PyInstaller update utilities.

Split from `vivarium/utils/` to separate deployment/process concerns from pure utilities.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Re-exports public API from all submodules |
| `paths.py` | Solid | Frozen vs dev mode detection, path resolution, command builders, `DEFAULT_JUPYTER_PORT` |
| `updater.py` | Solid (complex) | PyInstaller update checking, download, post-update merge |
| `_process.py` | Solid | PID lookup, kill, terminate — cross-platform process management |
| `_server.py` | Solid | gRPC server start/stop/wait |
| `_interface.py` | Solid | Panel interface start/stop, URL parsing from subprocess output, HTTP health checks |
| `_jupyter.py` | Solid | Jupyter server start/stop/check, port registry for cleanup tracking |
| `_ngrok.py` | Solid | Ngrok tunnel creation/teardown, Google Colab environment detection |
| `rthook_jupyter_matplotlib.py` | Solid | PyInstaller runtime hook for matplotlib/ipykernel compatibility |

## Public API

### From `__init__.py` (re-exports)

**Process management (`_process.py`):**
`kill_port_processes`, `kill_vivarium_processes`, `kill_all_vivarium_processes`, `stop_server_and_interface`

**Server (`_server.py`):**
`start_simulation_server`, `stop_simulation_server`, `wait_for_grpc_server`, `check_server_running`

**Interface (`_interface.py`):**
`wait_for_http`, `start_panel_interface`, `stop_panel_interface`

**Jupyter (`_jupyter.py`):**
`start_jupyter_server`, `stop_jupyter_server`, `check_jupyter_running`, `register_jupyter_port`, `unregister_jupyter_port`, `get_started_jupyter_ports`, `find_next_available_port`

**Ngrok (`_ngrok.py`):**
`create_ngrok_tunnel`, `close_ngrok_tunnel`, `check_colab_environment`, `get_ngrok_token`

**Paths (`paths.py`):**
`is_frozen`, `get_app_root`, `get_bundle_root`, `get_config_dir`, `get_notebooks_dir`, `get_defaults_dir`, `get_version`, `initialize_user_data`, `get_server_command`, `get_interface_command`, `get_jupyter_command`, `DEFAULT_JUPYTER_PORT`

### Internal API (not re-exported)

**`_process.py`:** `get_server_interface_pids`, `terminate_process`, `get_process_pids_unix`, `get_process_pids_windows` — used by `panel_app.py` directly.

## Internal Dependency Graph

```
paths.py (foundation — no runtime deps)
  ↑ updater.py
  ↑ _process.py
  ↑ _server.py ← _process.py
  ↑ _interface.py
  ↑ _jupyter.py ← _process.py (lazy import in stop_jupyter_server)
  ↑ _ngrok.py (standalone, no internal deps)
```

## Who Imports From Runtime

- **Scripts**: `run_server.py`, `run_interface.py`, `run_jupyter.py`
- **Controllers**: `vivarium_controller.py`
- **Interface**: `panel_app.py`
- **Simulator**: `simulator.py` (paths only)
- **Utils**: `scene_configs.py` (paths only)
- **Tests**: `conftest.py`, `test_vivarium_controller.py`, `test_start_stop_scripts.py`, `test_runtime.py`, `test_version.py`, `test_updater.py`
- **Notebooks**: `google_colab.ipynb`, `sandbox.ipynb`
- **CI**: `.github/workflows/build-release.yaml`

## Test Coverage

| Module | Test file | What's tested | Gaps |
|--------|-----------|---------------|------|
| `paths.py` | `test_runtime.py`, `test_version.py` | Path resolution, frozen mode paths, `initialize_user_data`, version file parsing | Command builders (`get_server_command`, etc.) untested |
| `updater.py` | `test_updater.py` | `check_for_updates`, `get_defaults_update_info`, `find_latest_backup_dir`, `perform_post_update_merge` | `download_update`, `apply_downloaded_update`, disk space check untested |
| `_server.py` | `test_start_stop_scripts.py` | Server start/stop via `start_simulation_server()` | Only happy path |
| `_process.py` | — | — | No dedicated tests |
| `_interface.py` | `test_start_stop_scripts.py` | Server+interface start/stop together | Minimal |
| `_jupyter.py` | — | — | No tests |
| `_ngrok.py` | — | — | No tests |

**Summary**: `paths.py` and `updater.py` have good coverage. Server start/stop has basic coverage. Process management, Jupyter, and ngrok are untested.
