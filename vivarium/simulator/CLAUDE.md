# vivarium/simulator/ — Package Audit

_Audited: 2026-03-10 | Status: Phase 1 Step 1_

## Purpose

Bridge between JAX-based simulation (`vivarium/environment/`) and client control (notebooks, Panel UI). Provides local API (`Simulator`) and remote API (gRPC server/client). Handles state serialization, controller parameter synchronization, and streaming.

## Files

| File | Size | Status | Purpose |
|------|------|--------|---------|
| `__init__.py` | 51B | Solid | Exports `Simulator` |
| `simulator.py` | 14K | Brittle | Core orchestrator: state management, run loop, controller parameters |
| `controller.py` | 586B | Solid | `SimulatorController` — wraps simulator controller_parameters for client-side access |
| `grpc_server/simulator_server.py` | 7.2K | Solid | gRPC servicer: RPC handlers, streaming |
| `grpc_server/simulator_client.py` | 12K | Solid | gRPC client: connects to server, streaming, bidirectional stepping |
| `grpc_server/converters.py` | 13K | Solid | Dataclass ↔ protobuf serialization |
| `grpc_server/simulator_pb2.py` | 9.0K | Generated | Protobuf Python code (do not edit) |
| `grpc_server/simulator_pb2_grpc.py` | 27K | Generated | gRPC Python code (do not edit) |
| `grpc_server/protos/simulator.proto` | 167L | Solid | Protocol buffer service definition |
| `grpc_server/numproto/numproto.py` | 956B | Solid | numpy array ↔ protobuf binary serialization |

## Architecture

### Shared Interface Pattern

`Simulator` and `SimulatorGRPCClient` expose the **same duck-typed interface** (`step()`, `get_state()`, `set_changes()`, `controller_parameters`, `remote`, `scene_name`, etc.). `VivariumController` takes either one as its `client` attribute:
- **`SimulatorGRPCClient`**: normal mode — communicates over gRPC to a separate server process
- **`Simulator`**: direct mode — no network, useful for testing and debugging

Distinguished by `client.is_grpc_client` (True/False), checked in `VivariumController` to decide whether to manage server lifecycle.

### Data Flow

```
Client (notebook/Panel)                    Server
─────────────────────                    ──────
SimulatorGRPCClient                      Simulator
  ├── state (numpy)         ←── gRPC ──   ├── state (JAX)
  ├── controller_parameters ←── gRPC ──   ├── controller_parameters
  └── step(changes=[...])   ─── gRPC ──→  ├── env.step(state) → new state
                                           └── Environment (JAX)
```

**Key points:**
- Server holds JAX state; client holds numpy copies (no JAX on client side)
- Full state transferred each time (no deltas)
- Changes sent as list of `(path, value)` tuples, applied via `update_dataclass_from_change_list()`
- Controller parameters built dynamically from config via `create_dataclass_from_dict()`

### gRPC RPCs

**Unary:** `Step`, `GetState`, `GetControllerParameters`, `GetStateAndControllerParameters`, `SetChanges`, `SetChangesAndStep`, `SetChangesReturnsState`, `GetSceneName`, `RegisterClient`, `UnregisterClient`, `Start`, `Stop`, `IsRunning`

**Streaming:** `StreamState` (server→client push), `BidirectionalStep` (two-way synchronized stepping)

**Dead:** `SetState` — handler calls undefined `simulator.set_state()` method. Never called by any client.

### Controller Parameters

Built in `Simulator.from_config()` by collecting `controller_kwargs` from simulator config and each component's client config. Result: `ControllerParameters(simulator=..., agents=..., walls=..., ...)`.

Synchronized: server centralizes updates → client fetches copy via RPC or bundled with state.

## Public API

### Simulator (server-side)
- `Simulator.from_config(config)` — create from Hydra config
- `step(changes=None)` — one JAX step, optionally apply changes first
- `run(threaded, num_steps, save)` — run loop (blocking or threaded)
- `stop()`, `is_running()`
- `get_state()`, `get_controller_parameters()`, `get_state_and_controller_parameters()`
- `set_changes(changes)` — apply changes to state/params
- `to_config(state)` — serialize to Hydra config
- `register_client(name)`, `unregister_client(name)`

### SimulatorGRPCClient (client-side)
- `SimulatorGRPCClient(name, server)` — connect to gRPC server
- `step(changes)`, `start()`, `stop()`, `is_running()`
- `get_state()`, `get_controller_parameters()`
- `set_changes(changes, update_from_server)`
- `start_state_stream(callback, max_fps)` / `stop_state_stream()` — async observation
- `bidirectional_step_sync(num_steps, compute_changes_fn)` — synchronized stepping
- `close()` — disconnect

### SimulatorController
- Wraps `controller_parameters.simulator` for client-side attribute access
- Used by `VivariumController` as `self.controllers['simulator']`

## Who Imports From Simulator

- **Scripts**: `run_server.py` — `Simulator.from_config()` + `serve()`
- **Controllers**: `vivarium_controller.py` — `SimulatorGRPCClient`, `SimulatorController`
- **Interface**: `panel_app.py` — `SimulatorGRPCClient`
- **Tests**: `conftest.py`, `test_simulator.py`, `test_grpc.py`, `test_edu_sessions.py`, `test_scene_config.py`
- **Dev scripts**: `benchmark_grpc.py`, `benchmark_streaming_real.py`

## Known Issues

### Should Fix

1. **`SetState` RPC handler is dead code.** `simulator_server.py:112` calls `self.simulator.set_state()` which doesn't exist on `Simulator`. Proto defines the RPC but no client ever calls it. Remove from proto, handler, and regenerate.

2. **Recording is broken.** `simulator.py:66` says "probably broken for now". `record()` appends JAX state to a list; `start_recording`/`stop_recording`/`load` exist but are untested and likely don't work. Either remove or fix.

3. **`nested_fields_to_access` dict is dead code.** `simulator.py:36-43`, marked "Now unused?". Also a commented-out decorator reference in `simulator_client.py:24`. Remove both.

4. **Dynamic attribute copying via `update_from_dataclass()`.** `simulator.py:236` copies fields like `run_from`, `simulation_running` from `controller_parameters.simulator` onto the `Simulator` instance. These attributes aren't declared in `__init__` and don't appear in type hints. Makes the code hard to understand and type-check.

### Medium Severity

5. **`StateAndControllerParameters` defined in two places.** As `@dataclass` in `simulator.py:45-48` and dynamically reconstructed in `simulator_client.py`. If they diverge, serialization breaks silently.

6. **`to_config()` uses fragile relative path.** Hardcoded `'../../conf/scene/simulator'` relative to module file. Should use `vivarium.utils.runtime` instead.

7. **Missing error handling in RPC handlers.** No try/catch — gRPC defaults to opaque 500 errors. Should log exceptions and return proper status codes.

8. **`scene_name` property setter raises `AttributeError`.** Unconventional — normally you just omit the setter. Minor code smell.

### Low Severity

9. **`load()` should be a standalone function, not a method.** Comment in code agrees: "TODO: This shouldn't be a method."

10. **Streaming doesn't respect `is_running` state.** `StreamState` yields even when simulator is paused, unlike `BidirectionalStep` which drives stepping.

## Test Coverage

| Area | Test File | Status |
|------|-----------|--------|
| `Simulator.from_config()`, `.step()`, `.to_config()` | `test_simulator.py` | Covered |
| State serialization (dataclass ↔ proto) | `test_grpc.py` | Covered |
| Controller parameters serialization | `test_grpc.py` | Covered |
| Changes serialization | `test_grpc.py` | Covered |
| Bidirectional streaming | `test_grpc.py` | Covered |
| `set_changes()` with update | `test_grpc.py` | Covered |
| Multi-client registration | `test_edu_sessions.py` | Covered |
| Recording (`start/stop/record/load`) | — | **Not tested** (broken feature) |
| `SetState` RPC | — | **Not tested** (dead code) |
| Streaming rate limiting | — | Not tested |

**Summary**: ~70% of live code is tested. Dead/broken features (recording, SetState) are untested, which is acceptable — they should be removed.

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Remove dead `SetState` RPC (proto + handler + regenerate) |
| High | Remove or fix recording feature |
| High | Remove `nested_fields_to_access` dead code |
| High | Make dynamic attributes explicit in `Simulator.__init__` |
| Medium | Centralize `StateAndControllerParameters` definition |
| Medium | Add error handling to RPC handlers |
| Medium | Fix `to_config()` path resolution |
| Low | Move `load()` out of Simulator class |
| Low | Simplify `scene_name` property |

## Structural Questions (resolved in Phase 1 Step 2)

1. **~~Is bidirectional streaming still needed?~~** Resolved: **no, it's dormant.** `bidirectional_step_sync` is implemented and tested, but no active code path uses it. Notebooks use unary RPCs via `controller.step()`. Panel UI uses periodic polling + one-way state streaming. The `use_streaming` flag in `base_interface.yaml` is vestigial (read but never passed to anything). `start_controller_thread(use_streaming=False)` defaults to unary. Only the benchmark script `dev/benchmark_streaming_real.py` calls it, and that script appears outdated. Candidate for removal to simplify the API surface.
