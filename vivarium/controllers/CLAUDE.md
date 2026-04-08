# vivarium/controllers/

## Purpose

Main user-facing API for programmatic control. `VivariumController` orchestrates client lifecycle, server management, stepping, and routing to component controllers. Also provides utilities for logging, routines, and behaviors. This is the primary entry point for notebook users.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Exports: `VivariumController`, `Controller`, `AttributeMapping` |
| `vivarium_controller.py` | Solid | Main controller: connection, stepping, interface lifecycle, component routing |
| `controller.py` | Solid | Base `Controller` class + `AttributeMapping` for remote attribute transforms |
| `handlers.py` | Solid | `Logger`, `RoutineHandler`, `BehaviorHandler` |
| `components/__init__.py` | Solid | Re-exports all component controllers (for PyInstaller discovery) |
| `components/entities/__init__.py` | Solid | Re-exports `EntityController`, `WallController`, `BraitenbergController` from environment |
| `components/physics/__init__.py` | Solid | Re-exports `CollisionController` |
| `components/eco_evo/__init__.py` | Solid | Re-exports `SpawnController`, `ConsumptionController` |

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
- `connect()` / `disconnect()` / `close()`
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

## Who Imports From Controllers

**Within vivarium:**
- `vivarium/simulator/controller.py` — `SimulatorController` inherits from `Controller`
- `vivarium/interface/panel_app.py` — creates `VivariumController`
- `vivarium/environment/components/*/controller.py` — import `Controller`, `AttributeMapping`, `RoutineHandler`, `Logger`, `BehaviorHandler`

**External:**
- Tests: `test_vivarium_controller.py`, `test_edu_sessions.py`, `conftest.py`
- All notebooks

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
| `Logger` | `test_edu_sessions/test_logger.py` | Covered (8 tests) |
| Error handling paths | — | **Not tested** |

**Summary**: Main flows well-tested. Error paths untested.

