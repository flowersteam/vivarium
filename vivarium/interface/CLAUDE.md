# vivarium/interface/

## Purpose

Panel web interface for the simulator. Provides a full UI with Bokeh visualization, simulation controls, component configuration tabs, Jupyter notebook embedding, and update management. Built on Panel + Bokeh.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `panel_app.py` | Brittle | `WindowManager`: handles scene selection, simulation UI, updates, Jupyter, and all callbacks |
| `parameterized.py` | Solid | `ParameterizedData`, `ParamSimulator`, `ParamEnvironment`: bi-directional param↔controller sync |
| `utils.py` | Solid (fragile pattern) | `cleanup_parameterized_class()`: cleans dynamic Param fields between instances |
| `jupyter_config_iframe.py` | Solid | Jupyter config for iframe embedding in Panel |
| `__init__.py` | — | Empty |

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

**Component interfaces** are loaded dynamically via Hydra: `hydra.utils.get_class(config.client.interface_cls)`. Interface classes live in `vivarium/components/*/interface.py`.

## Public API

- `WindowManager` — main entry point, instantiated by `scripts/run_interface.py`
- `create_interfaces(controllers, scene_config)` — factory function creating component interface instances
- `ParamSimulator`, `ParamEnvironment` — parameterized state sync objects

## Who Imports From Interface

- `scripts/run_interface.py` — `WindowManager`
- `tests/test_panel_app.py` — `WindowManager` (testing_mode)
- `tests/test_param.py` — `ParamSimulator`, `create_interfaces`

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

