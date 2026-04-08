# vivarium/simulator/

## Purpose

Bridge between JAX-based simulation (`vivarium/environment/`) and client control (notebooks, Panel UI). Provides local API (`Simulator`) and remote API (gRPC server/client). Handles state serialization, controller parameter synchronization, and streaming.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Exports `Simulator` |
| `simulator.py` | Brittle | Core orchestrator: state management, run loop, controller parameters |
| `controller.py` | Solid | `SimulatorController` — wraps simulator controller_parameters for client-side access |
| `grpc_server/simulator_server.py` | Solid | gRPC servicer: RPC handlers, streaming |
| `grpc_server/simulator_client.py` | Solid | gRPC client: connects to server, streaming, bidirectional stepping |
| `grpc_server/converters.py` | Solid | Dataclass ↔ protobuf serialization |
| `grpc_server/simulator_pb2.py` | Generated | Protobuf Python code (do not edit) |
| `grpc_server/simulator_pb2_grpc.py` | Generated | gRPC Python code (do not edit) |
| `grpc_server/protos/simulator.proto` | Solid | Protocol buffer service definition |
| `grpc_server/numproto/numproto.py` | Solid | numpy array ↔ protobuf binary serialization |

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

### Controller Parameters

Built in `Simulator.from_config()` by collecting `controller_kwargs` from simulator config and each component's client config. Result: `ControllerParameters(simulator=..., agents=..., walls=..., ...)`.

Synchronized: server centralizes updates → client fetches copy via RPC or bundled with state.

## Public API

### Simulator (server-side)
- `Simulator.from_config(config)` — create from Hydra config
- `step(changes=None)` — one JAX step, optionally apply changes first
- `run(threaded, num_steps)` — run loop (blocking or threaded)
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
| Streaming rate limiting | — | Not tested |

**Summary**: Core flows well-tested. Streaming rate limiting untested.

