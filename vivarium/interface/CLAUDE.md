# vivarium/interface/

## Purpose

Panel web interface for the simulator. Provides a full UI with Bokeh visualization, simulation controls, component configuration tabs, Jupyter notebook embedding, and update management. Built on Panel + Bokeh.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `panel_app.py` | Solid | `PanelApp`: scene selection, Bokeh plot, simulation controls, streaming, component tabs |
| `update_manager.py` | Solid | `UpdateManager`: update checking, downloading, defaults merge notifications |
| `jupyter_manager.py` | Solid | `JupyterManager`: Jupyter server lifecycle, notebook UI, conflict resolution |
| `parameterized.py` | Solid | `ParameterizedData`, `ParamSimulator`, `ParamEnvironment`: bi-directional param↔controller sync |
| `utils.py` | Solid (fragile pattern) | `cleanup_parameterized_class()`: cleans dynamic Param fields between instances |
| `jupyter_config_iframe.py` | Solid | Jupyter config for iframe embedding in Panel |
| `__init__.py` | — | Empty |

## Architecture

```
scripts/run_interface.py
  → PanelApp(controller=VivariumController)
    ├── Scene selection UI (if no controller provided)
    ├── UpdateManager — update check/download notifications
    ├── Simulation UI
    │   ├── Bokeh plot (entities via component Renderers)
    │   ├── Controls (start/stop, FPS, drag-drop)
    │   └── Component config tabs (via create_interfaces())
    └── JupyterManager — Jupyter server lifecycle, notebook iframe
```

**State sync loop** (periodic callback, default 33ms / ~30 FPS):
1. `update_plot_cb()` runs periodically
2. Calls `controller.apply_changes()` → sends UI changes to server (always, even if no new state)
3. Checks `_pending_state_update` flag — skips repaint if streaming is active but no new state arrived
4. Calls `renderer.update_cds(state)` for each component interface → updates Bokeh plot
5. ParamSimulator/ParamEnvironment handle slider↔controller sync via Param watchers

**Streaming** is always on in the Panel UI (no toggle). `_start_streaming()` is called unconditionally on connect. The `_pending_state_update` flag (set by the streaming callback, cleared by `update_plot_cb()`) prevents redundant repaints.

**Component interfaces** are loaded dynamically via Hydra: `hydra.utils.get_class(config.client.interface_cls)`. Interface classes live in `vivarium/components/*/interface.py`.

## Public API

- `PanelApp` — main entry point, instantiated by `scripts/run_interface.py`
- `UpdateManager` — update notification/download system (used by `PanelApp`)
- `JupyterManager` — Jupyter server lifecycle/UI (used by `PanelApp`)
- `create_interfaces(controllers, scene_config)` — factory function creating component interface instances
- `ParamSimulator`, `ParamEnvironment` — parameterized state sync objects

## Who Imports From Interface

- `scripts/run_interface.py` — `PanelApp`
- `tests/interface/test_panel_app.py` — `PanelApp` (testing_mode)
- `tests/interface/test_param.py` — `ParamSimulator`, `create_interfaces`

## Test Coverage

| Area | Test File | Status |
|------|-----------|--------|
| `ParamEntity`, `ParamSimulator`, entity params, collision params | `test_param.py` | Covered (4 tests) |
| `PanelApp` initialization (testing_mode) | `test_panel_app.py` | Minimal (1 test) |
| Scene selection UI | — | **Not tested** |
| `UpdateManager` | — | **Not tested** |
| `JupyterManager` | — | **Not tested** |
| State streaming | — | **Not tested** |
| All callbacks (start, FPS, drag-drop, theme) | — | **Not tested** |

**Summary**: `parameterized.py` is well-tested. `panel_app.py` has minimal test coverage — only basic initialization in testing_mode. `UpdateManager` and `JupyterManager` are now self-contained and independently testable, but have no tests yet.
