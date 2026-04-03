# tests/

## Purpose

Pytest test suite for the Vivarium project. 17 test files + conftest.py, ~3,150 LOC, ~98 tests. Mix of unit tests (~60%) and integration tests (~40%).

## Files

| File | Type | Tests | What It Tests |
|------|------|-------|---------------|
| `conftest.py` | Fixtures | — | 14+ fixtures: server lifecycle, gRPC, config loading, component factories |
| `test_edu_sessions.py` | Integration | 31 | Educational session scenarios via `vivarium_controller_start_session` |
| `test_update_check.py` | Unit | 13 | `vivarium.utils.updater` (mocked) |
| `test_vivarium_controller.py` | Integration | 11 | Controller connection, stepping, parameter access |
| `test_multi_spawn.py` | Unit+Integration | 13 | Spawn component (unit) + spawn controller (integration) |
| `test_components.py` | Unit | 7 | Component construction and stepping |
| `test_runtime.py` | Unit | 18 | Path resolution, frozen mode detection |
| `test_grpc.py` | Integration | 7 | gRPC serialization/deserialization |
| `test_dataclass_api.py` | Unit | 6 | `dataclass_wrapper.py`, gRPC converters |
| `test_param.py` | Integration | 3 | Parameterized UI integration |
| `test_version.py` | Unit | 6 | Version parsing |
| `test_scene_config.py` | Unit | 6 | Config loading, state creation |
| `test_environments.py` | Integration | 2 | Environment stepping (with NaN handling) |
| `test_start_stop_scripts.py` | Integration | 2 | Server/interface subprocess lifecycle |
| `test_simulator.py` | Integration | 3 | Simulator stepping |
| `test_state.py` | Unit | 1 | State initialization |
| `test_panel_app.py` | Integration | 1 | WindowManager initialization (testing_mode) |

## conftest.py Architecture

### Fixture Hierarchy

```
cleanup_vivarium_processes_session (session, autouse)
├── clean_server_state (function) — kills servers before subprocess tests
│   ├── server_fixture — subprocess server manager (.start/.stop)
│   └── server_and_interface_fixture — subprocess server + Panel interface
│
├── scene_config(scene_name) — factory: loads Hydra config
│   ├── environment_from_config — factory: creates Environment
│   ├── state_from_config — factory: creates state class
│   ├── simulator_from_config — factory: creates Simulator
│   │   ├── grpc_server — in-process gRPC server (fast, port 50051)
│   │   │   └── grpc_client — gRPC client connected to in-process server
│   │   │       └── vivarium_controller — wraps client in VivariumController
│   │   └── vivarium_controller_from_config — controller from config directly
│   └── controller_and_interfaces_from_config — controller + UI interfaces
│
├── cleanup_parameterized_class_fixture (function, autouse)
│
└── Component factory fixtures:
    step, braitenberg, spawn, proximity_map, consumption, energy, reproduction
    → environment(factories) → environment_and_state(factories)
```

### Key Patterns

- **Factory fixtures**: Most fixtures are factory functions (called with arguments), not direct values
- **Two server modes**: In-process gRPC server (fast, for unit-ish tests) vs subprocess server (slow, for lifecycle tests)
- **Component fixture chain**: `step → braitenberg → spawn → proximity_map → consumption → energy → reproduction`
- **Auto-cleanup**: Session-level process cleanup + per-test Param class cleanup

## Coverage Analysis

### Well-Tested

| Module | Test File | Coverage |
|--------|-----------|----------|
| `vivarium/controllers/vivarium_controller.py` | `test_vivarium_controller.py`, `test_edu_sessions.py` | Good (42 tests) |
| `vivarium/utils/runtime.py` | `test_runtime.py`, `test_version.py` | Good (24 tests) |
| `vivarium/utils/updater.py` | `test_update_check.py` | Good (13 tests, mocked) |
| `vivarium/utils/dataclass_wrapper.py` | `test_dataclass_api.py` | Good (6 tests) |
| `vivarium/utils/scene_configs.py` | `test_scene_config.py` | Good (6 tests) |
| `vivarium/simulator/grpc_server/` | `test_grpc.py` | Good (7 tests) |
| `vivarium/environment/components/eco_evo/spawn/` | `test_multi_spawn.py` | Good (13 tests) |
| `vivarium/interface/parameterized.py` | `test_param.py` | Adequate (3 tests) |

### Minimally Tested (indirect/smoke only)

| Module | Coverage |
|--------|----------|
| `vivarium/environment/environment.py` | 2 tests (stepping only) |
| `vivarium/environment/state.py` | 1 test |
| `vivarium/simulator/simulator.py` | 3 tests (stepping only) |
| `vivarium/interface/panel_app.py` | 1 test (init only) |
| `vivarium/utils/handle_server_interface.py` | 2 tests (start/stop only) |

### Not Tested

| Module | Notes |
|--------|-------|
| `vivarium/environment/render.py` | Has known bugs; no tests at all |
| `vivarium/environment/components/entities/braitenberg/sensorimotor.py` | Core sensing/motor logic |
| `vivarium/environment/components/entities/particle_lenia/` | Only via scene tests |
| `vivarium/environment/components/entities/walls/` | No tests |
| `vivarium/environment/components/physics/collision/` | No direct tests |
| `vivarium/environment/components/physics/friction/` | No tests |
| `vivarium/environment/components/physics/reset/` | No tests |
| `vivarium/environment/components/eco_evo/consumption/` | Indirect only (via test_components.py) |
| `vivarium/environment/components/eco_evo/energy/` | Indirect only |
| `vivarium/environment/components/eco_evo/reproduction/` | Known failure |
| `vivarium/controllers/controller.py` | Base class, no direct tests |
| `vivarium/controllers/utils.py` | Logger, RoutineHandler, BehaviorHandler untested |
| `vivarium/simulator/controller.py` | SimulatorController untested |
| `vivarium/utils/converters.py` | No tests |
| `vivarium/utils/jax_utils.py` | No tests |
| `vivarium/utils/timer.py` | No tests |
| All interface components | No tests |

## Structural Notes

- conftest.py fixture architecture supports both fast in-process and slow subprocess testing
- `test_edu_sessions.py` provides integration coverage of the student-facing API
- Component factory fixtures compose via the chain pattern
- Session-level cleanup prevents test pollution
- Test files are flat (no subdirectories mirroring source packages)
- No `@pytest.mark.slow` markers for selective test runs

