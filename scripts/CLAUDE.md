# scripts/ — Package Audit

_Audited: 2026-03-10 | Status: Phase 1 Step 1_

## Purpose

Entry points for the vivarium application. Three primary scripts (server, interface, combined launcher), plus PyInstaller support scripts, dev benchmarks, and two likely-dead utilities.

## Files

| File | Size | Status | Purpose |
|------|------|--------|---------|
| `run_server.py` | 1.5K | Solid | Start gRPC simulation server (Hydra config) |
| `run_interface.py` | 3.6K | Solid | Start Panel web interface with cleanup handlers |
| `run_vivarium.py` | 4.2K | Brittle | Combined server+interface launcher (PyInstaller main entry) |
| `run_jupyter.py` | 2.2K | Solid | Dual-mode Jupyter launcher (server or kernel) for PyInstaller |
| `patch_jax_md.py` | 2.0K | Solid | Patch jax_md for JAX 0.4.24+ compat (Intel Mac) |
| `print_config.py` | 251B | Dead code | Prints Hydra config — unused, undocumented |
| `profiling.py` | 420B | Incomplete | JAX profiler — hardcoded scene, uses deprecated imports |
| `rthook_jupyter_matplotlib.py` | 2.7K | Solid | PyInstaller runtime hook for matplotlib/ipykernel |
| `dev/benchmark_grpc.py` | 13K | Solid | gRPC communication pattern benchmarks |
| `dev/benchmark_streaming_real.py` | 5.3K | Solid | Real-world streaming benchmark with behaviors |

## Invocation Patterns

### User-facing CLI
```
python scripts/run_server.py scene=<name>        # Start server
python scripts/run_interface.py [--flags]          # Start web UI
python scripts/run_vivarium.py [scene] [--no-browser]  # Combined
python scripts/patch_jax_md.py                     # Post-install fix
```

### Programmatic (via `vivarium.utils.handle_server_interface`)
- `start_simulation_server()` → spawns `run_server.py` as subprocess
- `start_panel_interface()` → spawns `run_interface.py` as subprocess
- In frozen mode, spawns compiled executables instead (via `runtime.py` abstraction)

### PyInstaller
- `run_server.py` and `run_interface.py` bundled as separate executables in `.spec`
- `rthook_jupyter_matplotlib.py` loaded as runtime hook automatically
- `run_jupyter.py` used for embedded Jupyter

## Dependencies

All primary scripts depend on `vivarium.utils.runtime` for frozen/dev mode detection. `run_server.py` depends on `vivarium.simulator`. `run_interface.py` depends on `vivarium.interface` and `panel`. Benchmark scripts depend on internal gRPC APIs.

## Known Issues

### Should Fix

1. **`print_config.py` — dead code.** Remove it.

2. **`profiling.py` — incomplete.** Hardcoded to braitenberg scene, uses deprecated `SceneConfiguration` import, hardcoded `/tmp/tensorboard` path. Either complete it (parameterize, update imports) or remove.

3. **`run_vivarium.py` — purpose unclear, possibly dead.** Originally launched server + interface on a given scene in one command. Since then, `run_interface.py` has been extended to handle scene selection from the UI, server startup, and interface launch — making `run_vivarium.py` redundant. Also has over-defensive PyInstaller guards (spawn counter, unused env vars). Candidate for removal — discuss in Step 3 (planning discussion).

### Minor

4. **`rthook_jupyter_matplotlib.py` location.** Should live in a `pyinstaller_hooks/` directory per convention, not in `scripts/`.

5. **Inconsistent scene argument style.** `run_server.py` uses Hydra override (`scene=name`), `run_vivarium.py` uses positional arg. Minor UX inconsistency.

6. **No `--version` flag** on any script.

7. **Global state in `run_interface.py`** (`_cleanup_done`, `_window_manager`). Works but inelegant.

## Test Coverage

| Script | Test file | What's tested | Gaps |
|--------|-----------|---------------|------|
| `run_server.py` | `test_start_stop_scripts.py` | Server start/stop via `start_simulation_server()` | No direct invocation test; relies on `handle_server_interface` wrapper |
| `run_interface.py` | `test_start_stop_scripts.py` | Server+interface start/stop together | Cleanup handlers, signal handling untested |
| `run_vivarium.py` | — | — | No tests. PyInstaller guards, spawn counter, combined launch flow all untested |
| `run_jupyter.py` | — | — | No tests. Dual-mode detection untested |
| `patch_jax_md.py` | — | — | No tests (acceptable — manual post-install tool) |
| `print_config.py` | — | — | Dead code, no tests needed |
| `profiling.py` | — | — | Incomplete, no tests needed until fixed |
| `rthook_jupyter_matplotlib.py` | — | — | No tests (PyInstaller runtime hook, hard to unit test) |
| `dev/benchmark_*.py` | — | — | Dev tools, no tests expected |

**Summary**: Only `run_server.py` and `run_interface.py` have indirect test coverage via `test_start_stop_scripts.py`. `run_vivarium.py` — the most complex script with brittle PyInstaller guards — has no tests at all.

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Remove `print_config.py` (dead code) |
| High | Remove or complete `profiling.py` |
| Medium | Simplify `run_vivarium.py` PyInstaller guards |
| Medium | Move `rthook_jupyter_matplotlib.py` to `pyinstaller_hooks/` |
| Low | Add `--version` flag to entry point scripts |
| Low | Standardize scene argument style across scripts |
