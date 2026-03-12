# vivarium/controllers/ — Package Audit

_Audited: 2026-03-10 | Status: Phase 1 Step 1_

## Purpose

Main user-facing API for programmatic control. `VivariumController` orchestrates client lifecycle, server management, stepping, and routing to component controllers. Also provides utilities for logging, routines, and behaviors. This is the primary entry point for notebook users.

## Files

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `__init__.py` | 3 | Solid | Exports: `VivariumController`, `Controller`, `AttributeMapping`, `set_nested_attr`, `kill_session` |
| `vivarium_controller.py` | 572 | Solid | Main controller: connection, stepping, interface lifecycle, component routing |
| `controller.py` | 71 | Solid | Base `Controller` class + `AttributeMapping` for remote attribute transforms |
| `utils.py` | 333 | Solid | `Logger`, `RoutineHandler`, `BehaviorHandler`, `kill_session()` |
| `components/__init__.py` | 10 | Solid | Re-exports all component controllers (for PyInstaller discovery) |
| `components/entities/__init__.py` | 6 | Solid | Re-exports `EntityController`, `WallController`, `BraitenbergController` from environment |
| `components/physics/__init__.py` | 2 | Solid | Re-exports `CollisionController` |
| `components/eco_evo/__init__.py` | 5 | Solid | Re-exports `SpawnController`, `ConsumptionController` |

## Architecture

### Controller Hierarchy

```
VivariumController
  ├── client (Simulator or SimulatorGRPCClient — shared interface)
  ├── controllers = {
  │     'simulator':   SimulatorController         (from simulator/controller.py)
  │     'agents':      EntityListController         (from environment/components/entities/)
  │     'objects':     EntityListController
  │     'collision':   CollisionController          (from environment/components/physics/)
  │     'spawn':      SpawnController               (from environment/components/eco_evo/)
  │     'consumption': ConsumptionController
  │     ...
  │   }
  ├── routine_handler (RoutineHandler)
  └── logger (Logger)
```

Controllers are instantiated dynamically from Hydra config via `hydra.utils.get_class(c_config.client.controller_cls)`.

### Key Patterns

**Remote proxy for deferred changes:** All controllers manipulate `client.remote` (a `Remote` proxy from `dataclass_wrapper.py`). Setting attributes records changes; `apply_changes()` batches them and sends to server.

**Attribute mapping:** `Controller` base class supports `AttributeMapping` to transform between user-friendly property names (`.color`, `.subtype`) and internal state field names/types.

**Step routing:** `VivariumController.step()` calls `controller_step()` on all component controllers, then calls `simulator_step()` if this client is the designated stepper (`simulator.run_from == client.name`).

**Shared subtype labels:** A single list object is shared by reference across all controllers. `set_subtype_labels()` mutates it in-place so all controllers see the change immediately.

**`components/` subpackages** are pure re-exports from `vivarium/environment/components/*/controller.py`. They exist for clean import paths and PyInstaller discovery.

## Public API (key methods)

### VivariumController
- `start_session(scene_name, ...)` — classmethod: start server + connect + configure
- `connect()` / `disconnect()` / `close()` (`close_session()` is candidate for removal)
- `step()` — one step: run component controllers then simulator step
- `start_controller_thread()` / `stop_controller_thread()`
- `start_interface()` / `stop_interface()`
- `attach_routine(fn, interval)` / `detach_routine(fn)`
- `set_subtype_labels(new_labels)`
- Dynamic attribute access: `controller.agents`, `controller.spawn`, etc. → delegates to `self.controllers[name]`

### Controller (base class)
- `from_config(name, remote)` — instantiate from Hydra config
- `step(time, catch_errors)` — called by VivariumController each tick
- `remote_to_ctrl(attr)` / `ctrl_to_remote(attr, value)` — attribute transformation

### Utilities
- `Logger` — simple dict-of-lists for recording data during simulation
- `RoutineHandler` — attach/detach callback functions called each N steps
- `BehaviorHandler` — like RoutineHandler but with weighted motor output blending (for Braitenberg agents)
- `kill_session(global_vars)` — safely close controller and stop IPython kernel

## Who Imports From Controllers

**Within vivarium:**
- `vivarium/simulator/controller.py` — `SimulatorController` inherits from `Controller`
- `vivarium/interface/panel_app.py` — creates `VivariumController`
- `vivarium/environment/components/*/controller.py` — import `Controller`, `AttributeMapping`, `RoutineHandler`, `Logger`, `BehaviorHandler`

**External:**
- Tests: `test_vivarium_controller.py`, `test_edu_sessions.py`, `conftest.py`
- All notebooks

## Known Issues

### Should Fix

1. **`__init__.py` exports questionable symbols.** `set_nested_attr` is only used internally by `Controller.__setattr__` — no external consumer. `Controller` and `AttributeMapping` are imported directly from `controller.py` by all consumers, not via `__init__`. Consider cleaning up exports to just `VivariumController`.

2. **`kill_session()` is likely dead code.** Only referenced in `session_6_bonus.ipynb` (outdated) and `copilot-instructions.md`. No active notebook uses it. Consider removing.

3. **Inconsistent import style in environment component controllers.** Some use deep relative imports (`from .....controllers.controller`), others use absolute (`from vivarium.controllers`). Should standardize on absolute imports.

4. **`apply_changes()` has a TODO** questioning whether it belongs here or in the client. Duplicates some logic from `SimulatorGRPCClient.set_changes()`. Clarify ownership.

### Medium Severity

3. **`RoutineHandler` and `BehaviorHandler` are similar but separate.** Both implement attach/detach/step patterns with intervals. `BehaviorHandler` adds weighted blending. Could share a base class, but may not be worth the refactor effort.

4. **Step routing logic is implicit.** Whether this client drives stepping depends on `simulator.run_from == client.name`, set via `start_session()` parameters. Not documented clearly.

5. **`BehaviorHandler.behave()` assumes agent is an `EntityController`** with `__setattr__` recording. Tight coupling, undocumented contract.

### Low Severity

6. **Missing docstrings** on `Controller.to_deal_with()`, `Controller.remote_to_ctrl()`, `Controller.ctrl_to_remote()`.

7. **`kill_session()` touches IPython internals** (`IPython.Application.instance().kernel.do_shutdown()`). Fragile but acceptable for a notebook utility.

## Test Coverage

| Area | Test File | Status |
|------|-----------|--------|
| Constructor (all modes) | `test_vivarium_controller.py` | Covered (7 tests) |
| Connection/disconnection | `test_vivarium_controller.py` | Covered (5 tests) |
| Server lifecycle | `test_vivarium_controller.py` | Covered (3 tests) |
| Parameter access & mutation | `test_vivarium_controller.py` | Covered |
| Controller parameter sync (multi-client) | `test_vivarium_controller.py` | Covered |
| `set_subtype_labels()` | `test_edu_sessions.py` | Covered (7 tests) |
| Component controller instantiation | `test_vivarium_controller.py` | Indirect |
| Routines & behaviors | `test_edu_sessions.py` | Indirect |
| `Logger` | — | **Not tested** |
| `kill_session()` | — | **Not tested** |
| Error handling paths | — | **Not tested** |

**Summary**: Main flows well-tested. Utility classes (`Logger`, `kill_session`) and error paths untested.

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Standardize imports to absolute paths in environment component controllers |
| Medium | Document step routing / run_from state machine |
| Medium | Add docstrings to Controller base class methods |
| Medium | Clarify apply_changes() ownership (controller vs client) |
| Low | Add type hints to Controller, RoutineHandler, BehaviorHandler |
| Low | Test Logger and kill_session() |

## Structural Questions (updated in Phase 1 Step 2)

1. **Component controllers live in the wrong package.** Not a circular import issue (Python imports work fine), but a design flaw. Component controllers (`BraitenbergController`, `SpawnController`, etc.) and interfaces are *defined inside* `environment/components/*/controller.py`, yet they are client-side code that imports base classes from `vivarium/controllers/` and `vivarium/interface/`. This means the environment package — which should be a pure simulation engine — contains its own abstraction layer and depends on packages above it in the architecture.

   **Proposed fix:** Move `vivarium/environment/components/` to `vivarium/components/` as a top-level package. This works because `environment.py` resolves component classes dynamically via `hydra.utils.get_class(_target_)` — no static imports from components. The result:
   - `environment/` becomes a pure JAX engine (state, orchestrator, math)
   - `components/` is explicitly the cross-cutting "glue" package (JAX + controller + interface per feature)
   - `controllers/components/` and `interface/components/` re-export shims become unnecessary
   - All dependency arrows point in sensible directions

   **Cost:** Mechanical but significant — every `vivarium.environment.components` import and every `_target_` string in YAML configs must be updated. Evaluate for Phase 2.
