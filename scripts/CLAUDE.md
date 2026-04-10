# scripts/

## Purpose

Entry points for the vivarium application. Three primary scripts (server, interface, combined launcher), plus PyInstaller support scripts and dev benchmarks.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `run_server.py` | Solid | Start gRPC simulation server (Hydra config) |
| `run_interface.py` | Solid | Start Panel web interface with cleanup handlers |
| `run_jupyter.py` | Solid | Dual-mode Jupyter launcher (server or kernel) for PyInstaller |
| `patch_jax_md.py` | Solid | Patch jax_md for JAX 0.4.24+ compat (Intel Mac) |
| `print_config.py` | Unused | Prints Hydra config |
| `profiling.py` | Solid | JAX profiler — profiles braitenberg scene step |
| `dev/benchmark_grpc.py` | Solid | gRPC communication pattern benchmarks |

## Invocation Patterns

### User-facing CLI
```
python scripts/run_server.py scene=<name>        # Start server
python scripts/run_interface.py [--flags]          # Start web UI
python scripts/patch_jax_md.py                     # Post-install fix
```

### Programmatic (via `vivarium.runtime`)
- `start_simulation_server()` -> spawns `run_server.py` as subprocess
- `start_panel_interface()` -> spawns `run_interface.py` as subprocess
- In frozen mode, spawns compiled executables instead (via `vivarium.runtime.paths` abstraction)

### PyInstaller
- `run_server.py` and `run_interface.py` bundled as separate executables in `.spec`
- `vivarium/runtime/rthook_jupyter_matplotlib.py` loaded as runtime hook automatically
- `run_jupyter.py` used for embedded Jupyter

## Dependencies

All primary scripts depend on `vivarium.runtime.paths` for frozen/dev mode detection. `run_server.py` depends on `vivarium.simulator`. `run_interface.py` depends on `vivarium.interface` and `panel`. Benchmark scripts depend on internal gRPC APIs.

## Test Coverage

| Script | Test file | What's tested | Gaps |
|--------|-----------|---------------|------|
| `run_server.py` | `test_start_stop_scripts.py` | Server start/stop via `start_simulation_server()` | No direct invocation test; relies on `vivarium.runtime` wrapper |
| `run_interface.py` | `test_start_stop_scripts.py` | Server+interface start/stop together | Cleanup handlers, signal handling untested |
| `run_jupyter.py` | — | — | No tests. Dual-mode detection untested |
| `patch_jax_md.py` | — | — | No tests (acceptable — manual post-install tool) |
| `print_config.py` | — | — | No tests |
| `profiling.py` | — | — | No tests |
| `dev/benchmark_*.py` | — | — | Dev tools, no tests expected |

**Summary**: Only `run_server.py` and `run_interface.py` have indirect test coverage via `test_start_stop_scripts.py`. Other scripts have no tests.
