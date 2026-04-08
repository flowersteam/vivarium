# tests/

## Purpose

Pytest test suite for the Vivarium project. Structure mirrors source packages.

## Structure

```
tests/
  conftest.py                        # root: shared fixtures (server lifecycle, gRPC, config, cleanup)
  __init__.py
  environment/
    __init__.py
    conftest.py                      # component fixture chain (step → braitenberg → ...)
    test_environment.py              # Environment stepping
    test_state.py                    # State initialization
    test_components.py               # Component construction and stepping
    test_multi_spawn.py              # Spawn component + controller
  simulator/
    __init__.py
    test_simulator.py                # Simulator stepping
    test_grpc.py                     # gRPC serialization/deserialization
  controllers/
    __init__.py
    test_vivarium_controller.py      # Controller connection, stepping, parameters
    test_edu_sessions/               # Educational session integration tests
      __init__.py
      conftest.py                    # controller/running_controller fixtures
      test_subtype_labels.py
      test_entity_access.py
      test_properties.py
      test_sensing.py
      test_behaviors.py
      test_routines.py
      test_logger.py
      test_consumption_spawn.py
      test_internal_state.py
  interface/
    __init__.py
    test_panel_app.py                # WindowManager initialization
    test_param.py                    # Parameterized UI integration
  utils/
    __init__.py
    test_runtime.py                  # Path resolution, frozen mode detection
    test_version.py                  # Version parsing
    test_scene_config.py             # Config loading, state creation
    test_updater.py                  # vivarium.utils.updater (mocked)
    test_dataclass_wrapper.py        # dataclass_wrapper.py, gRPC converters
  scripts/
    __init__.py
    test_start_stop_scripts.py       # Server/interface subprocess lifecycle
```

## conftest.py Architecture

### Root conftest.py — Shared fixtures

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
│   │   │       └── vivarium_controller_start_session — controller via start_session
│   │   └── vivarium_controller_from_config — controller from config directly
│   └── controller_and_interfaces_from_config — controller + UI interfaces
│
├── vivarium_controller — factory: wraps any client in VivariumController
│
└── cleanup_parameterized_class_fixture (function, autouse)
```

### environment/conftest.py — Component fixture chain

```
step → braitenberg → spawn
                   → proximity_map → consumption → energy → reproduction
→ environment(factories) → environment_and_state(factories)
```

### Key Patterns

- **Factory fixtures**: Most fixtures are factory functions (called with arguments), not direct values
- **Two server modes**: In-process gRPC server (fast, for unit-ish tests) vs subprocess server (slow, for lifecycle tests)
- **Component fixture chain**: Composable component lists in `environment/conftest.py`
- **Auto-cleanup**: Session-level process cleanup + per-test Param class cleanup
- **`@pytest.mark.slow`**: Marks tests using subprocess or gRPC (registered in `.pytest.ini`)

## Coverage Gaps

Test directory mirrors source structure, so mapping is self-evident. Below lists modules with no or minimal dedicated tests:

| Module | Notes |
|--------|-------|
| `vivarium/environment/render.py` | Has known bugs; no tests |
| `vivarium/components/entities/braitenberg/sensorimotor.py` | Core sensing/motor logic, no direct tests |
| `vivarium/components/entities/particle_lenia/` | Only via scene tests |
| `vivarium/components/entities/walls/` | No tests |
| `vivarium/components/physics/collision/` | No direct tests |
| `vivarium/components/physics/friction/` | No tests |
| `vivarium/components/physics/reset/` | No tests |
| `vivarium/components/eco_evo/consumption/` | Indirect only (via test_components.py) |
| `vivarium/components/eco_evo/energy/` | Indirect only |
| `vivarium/components/eco_evo/reproduction/` | Indirect only |
| `vivarium/controllers/controller.py` | Base class, no direct tests |
| `vivarium/controllers/handlers.py` | RoutineHandler, BehaviorHandler tested indirectly via test_routines.py / test_behaviors.py |
| `vivarium/simulator/controller.py` | SimulatorController untested |
| `vivarium/utils/converters.py` | No tests |
| `vivarium/utils/jax_utils.py` | No tests |
| `vivarium/utils/timer.py` | No tests |
| All interface components | No tests |
