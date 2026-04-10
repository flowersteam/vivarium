# vivarium/simulator/

## Purpose

Bridge between JAX-based simulation (`vivarium/environment/`) and client control (notebooks, Panel UI). Provides local API (`Simulator`) and remote API (gRPC server/client). Handles state serialization, controller parameter synchronization, and streaming.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `__init__.py` | Solid | Exports `Simulator` |
| `simulator.py` | Solid | Core orchestrator: state management, run loop, controller parameters |
| `controller.py` | Solid | `SimulatorController` — wraps simulator controller_parameters for client-side access |
| `grpc_server/simulator_server.py` | Solid | gRPC servicer: RPC handlers, streaming |
| `grpc_server/simulator_client.py` | Solid | gRPC client: connects to server, state streaming |
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

**Streaming:** `StreamState` (server→client push)

### Controller Parameters

Built in `Simulator.from_config()` by collecting `controller_kwargs` from simulator config and each component's client config. Result: `ControllerParameters(simulator=..., agents=..., walls=..., ...)`.

Synchronized: server centralizes updates → client fetches copy via RPC or bundled with state.

### Single Source of Truth: `controller_parameters.simulator`

`controller_parameters.simulator` is the single source of truth for shared config fields (`freq`, `scene_name`, `run_from`, `simulation_running`, `client_names`, `subtype_labels`, `env.*`). Simulator never stores these as instance attributes.

- **`__init__`**: builds a default `controller_parameters` when none is provided (headless path: `Simulator(env=env, freq=10)`), so both headless and config paths are uniform.
- **`__setattr__` safeguard**: blocks creation of new instance attributes that shadow fields on `controller_parameters.simulator`, preventing ghost attribute regression. Existing instance attributes (e.g. `self.env`) and properties (e.g. `scene_name`) are allowed through.
- **`set_changes()`**: applies changes via `update_dataclass_from_change_list`, then explicitly syncs derived state: `sleep_timer.frequency` from `freq`, and env config fields (box_size, etc.) to the actual `Environment` via `sync_dataclass_fields()`.
- **`scene_name`**: read-only property delegating to `controller_parameters.simulator.scene_name` (part of duck-typed interface with `SimulatorGRPCClient`).
- **`freq`**: no property — accessed as `controller_parameters.simulator.freq` everywhere. Clients access it via `SimulatorController`, not directly on Simulator.

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
- `close()` — disconnect

### SimulatorController
- Wraps `controller_parameters.simulator` for client-side attribute access
- Used by `VivariumController` as `self.controllers['simulator']`

## Who Imports From Simulator

- **Scripts**: `run_server.py` — `Simulator.from_config()` + `serve()`
- **Controllers**: `vivarium_controller.py` — `SimulatorGRPCClient`, `SimulatorController`
- **Interface**: `panel_app.py` — `SimulatorGRPCClient`
- **Tests**: `conftest.py`, `test_simulator.py`, `test_grpc.py`, `test_edu_sessions.py`, `test_scene_config.py`
- **Dev scripts**: `benchmark_grpc.py`

## Test Coverage

| Area | Test File | Status |
|------|-----------|--------|
| `Simulator.from_config()`, `.step()`, `.to_config()` | `test_simulator.py` | Covered |
| State serialization (dataclass ↔ proto) | `test_grpc.py` | Covered |
| Controller parameters serialization | `test_grpc.py` | Covered |
| Changes serialization | `test_grpc.py` | Covered |
| `set_changes()` with update | `test_grpc.py` | Covered |
| `__setattr__` ghost attribute safeguard | `test_simulator.py` | Covered |
| Multi-client registration | `test_edu_sessions.py` | Covered |
| Streaming rate limiting | — | Not tested |

**Summary**: Core flows well-tested. Streaming rate limiting untested.

