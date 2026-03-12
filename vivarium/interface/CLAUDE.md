# vivarium/interface/ — Package Audit

_Audited: 2026-03-11 | Status: Phase 1 Step 1_

## Purpose

Panel web interface for the simulator. Provides a full UI with Bokeh visualization, simulation controls, component configuration tabs, Jupyter notebook embedding, and update management. Built on Panel + Bokeh.

## Files

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `panel_app.py` | 1410 | Brittle | `WindowManager`: monolithic god-object handling all UI concerns |
| `parameterized.py` | 113 | Solid | `ParameterizedData`, `ParamSimulator`, `ParamEnvironment`: bi-directional param↔controller sync |
| `utils.py` | 22 | Solid (fragile pattern) | `cleanup_parameterized_class()`: cleans dynamic Param fields between instances |
| `jupyter_config_iframe.py` | 26 | Solid | Jupyter config for iframe embedding in Panel |
| `__init__.py` | 0 | — | Empty |
| `components/` | 4 files | Solid | Re-export shims for PyInstaller discovery of environment component interfaces |

## Architecture

```
scripts/run_interface.py
  → WindowManager(controller=VivariumController)
    ├── Scene selection UI (if no controller provided)
    ├── Simulation UI
    │   ├── Bokeh plot (entities via component Renderers)
    │   ├── Controls (start/stop, FPS, drag-drop)
    │   └── Component config tabs (via create_interfaces())
    ├── Jupyter notebook iframe
    └── Update check/download system
```

**State sync loop** (periodic callback, default 33ms / ~30 FPS):
1. `update_plot_cb()` runs periodically
2. Calls `renderer.update_cds(state)` for each component interface → updates Bokeh plot
3. Calls `controller.apply_changes()` → sends UI changes to server
4. ParamSimulator/ParamEnvironment handle slider↔controller sync via Param watchers

**Component interfaces** are loaded dynamically via Hydra: `hydra.utils.get_class(config.client.interface_cls)`. The `components/` subpackage re-exports them for PyInstaller static analysis.

## Public API

- `WindowManager` — main entry point, instantiated by `scripts/run_interface.py`
- `create_interfaces(controllers, scene_config)` — factory function creating component interface instances
- `ParamSimulator`, `ParamEnvironment` — parameterized state sync objects

## Who Imports From Interface

- `scripts/run_interface.py` — `WindowManager`
- `tests/test_panel_app.py` — `WindowManager` (testing_mode)
- `tests/test_param.py` — `ParamSimulator`, `create_interfaces`

## Known Issues

### Should Fix

1. **`panel_app.py` is a 1410-line monolith.** `WindowManager` handles scene selection, simulation UI, update checking/downloading, Jupyter server management, and all callbacks. Should be split — at minimum extract `UpdateManager` and `JupyterManager`.

2. **State streaming is ineffective.** `_on_state_stream_update()` sets a `_pending_state_update` threading.Event, but `update_plot_cb()` never checks it. The periodic callback polls at 33ms regardless of streaming. Streaming doesn't reduce polling.

3. **Dead code:**
   - `self.streaming_toggle` widget created (line 687) but never displayed — commented out at line 1368
   - `streaming_toggle_cb()` defined (line 972) but never registered — commented out at line 1391
   - `self.notebook_mode` parameter set (line 84) but never read anywhere
   - Lines 993-994: commented-out config_update logic

4. **Typo: `udpate_other_interfaces`** called at line 645 (with TODO comment acknowledging it). Same typo as in the base class in `environment/components/interface.py`.

### Medium Severity

5. **Dynamic Param field cleanup is fragile** (`utils.py`). Only handles `ParamAgent` and `ParamParticleLenia`. Adding new entity types requires manually updating the cleanup dict. Uses private Param API.

6. **Missing error logging.** `_start_server_cb()` catches exceptions but only updates status UI — no `lg.exception()` for stack traces.

7. **Thread safety concerns.** Multiple callbacks can race on state updates. `update_plot_cb()` accesses `controller.client.state` directly with no locking.

8. **Jupyter port hardcoded** at 8889 (line 729). Should be configurable via scene config.

### Low Severity

9. **Empty `__init__.py`** — no `__all__`, public API undiscoverable.

10. **Missing docstrings** on `WindowManager.__init__()`, `create_interfaces()`, `ParameterizedData` methods.

11. **`parameterized.py` has a TODO** (line 6) signaling intent to rename the file.

## Test Coverage

| Area | Test File | Status |
|------|-----------|--------|
| `ParamEntity`, `ParamSimulator`, entity params, collision params | `test_param.py` | Covered (4 tests) |
| `WindowManager` initialization (testing_mode) | `test_panel_app.py` | Minimal (1 test) |
| Scene selection UI | — | **Not tested** |
| Update system | — | **Not tested** |
| Jupyter integration | — | **Not tested** |
| State streaming | — | **Not tested** |
| All callbacks (start, FPS, drag-drop, theme) | — | **Not tested** |

**Summary**: `parameterized.py` is well-tested. `panel_app.py` has minimal test coverage — only basic initialization in testing_mode. All interactive features (callbacks, streaming, Jupyter, updates) are untested.

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Remove dead code (streaming_toggle, notebook_mode, commented lines) |
| High | Fix typo `udpate_other_interfaces` → `update_other_interfaces` (here and in environment base class) |
| High | Verify state streaming effectiveness (may actually work — needs investigation) |
| High | Rename `WindowManager` — misleading name, doesn't manage windows |
| Medium | Extract UpdateManager class from WindowManager |
| Medium | Extract JupyterManager class from WindowManager |
| Medium | Add error logging (`lg.exception()`) to callback error handlers |
| Low | Add `__all__` to `__init__.py` |
| Low | Add docstrings to public API |
| Low | Make Jupyter port configurable |

## Structural Questions (updated in Phase 1 Step 2)

1. **Could Bokeh renderers replace `render.py`?** Deferred to Phase 2/3. The Bokeh renderers know how to visualize entities but launching them headlessly is non-trivial. `render.py` stays in `environment/` for now (see environment CLAUDE.md).

2. **State streaming.** Cross-package investigation confirmed: the `_pending_state_update` flag is set but never checked by `update_plot_cb()`. However, the stream callback updates `self.controller.client.state` directly, which `update_plot_cb()` reads. Streaming may work via this side-effect path. Needs testing before removing — but the `use_streaming` config flag is vestigial (never consumed). See also: `bidirectional_step_sync` is dormant (simulator CLAUDE.md).

3. **`WindowManager` naming.** Still a bad name. Rename to `SimulationApp` or similar during Phase 2 cleanup.

4. **Param cleanup pattern.** Still fragile. Defer evaluation to Phase 2 — low risk for release if no new entity types are added.
