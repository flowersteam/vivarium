# tests/ — Package Audit

_Audited: 2026-03-11 | Status: Phase 1 Step 1_

## Purpose

Pytest test suite for the Vivarium project. 17 test files + conftest.py, ~3,150 LOC, ~98 tests. Mix of unit tests (~60%) and integration tests (~40%).

## Files

| File | LOC | Type | Tests | What It Tests |
|------|-----|------|-------|---------------|
| `conftest.py` | 423 | Fixtures | — | 14+ fixtures: server lifecycle, gRPC, config loading, component factories |
| `test_edu_sessions.py` | 422 | Integration | 31 | Educational session scenarios via `vivarium_controller_start_session` |
| `test_update_check.py` | 546 | Unit | 13 | `vivarium.utils.updater` (mocked) |
| `test_vivarium_controller.py` | 361 | Integration | 11 | Controller connection, stepping, parameter access |
| `test_multi_spawn.py` | 256 | Unit+Integration | 13 | Spawn component (unit) + spawn controller (integration) |
| `test_components.py` | 214 | Unit | 7 | Component construction and stepping |
| `test_runtime.py` | 190 | Unit | 18 | Path resolution, frozen mode detection |
| `test_grpc.py` | 153 | Integration | 7 | gRPC serialization/deserialization |
| `test_dataclass_api.py` | 137 | Unit | 6 | `dataclass_wrapper.py`, gRPC converters |
| `test_param.py` | 103 | Integration | 3 | Parameterized UI integration |
| `test_version.py` | 68 | Unit | 6 | Version parsing |
| `test_scene_config.py` | 63 | Unit | 6 | Config loading, state creation |
| `test_environments.py` | 62 | Integration | 2 | Environment stepping (with NaN handling) |
| `test_start_stop_scripts.py` | 48 | Integration | 2 | Server/interface subprocess lifecycle |
| `test_simulator.py` | 40 | Integration | 3 | Simulator stepping |
| `test_state.py` | 38 | Unit | 1 | State initialization |
| `test_panel_app.py` | 22 | Integration | 1 | WindowManager initialization (testing_mode) |

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

## Known Issues

### Should Fix

1. **No-op tests with no assertions.** `test_components.py:test_instantiate()` has just `pass` — verifies nothing. `test_braitenberg()` steps the environment but only has `pass` at the end — pure smoke test.

2. **Duplicate test function names.** `test_dataclass_api.py` has two functions named `test_remote()` (lines ~56 and ~87). Both run but second shadows first in reporting. Should be renamed.

3. **Incomplete tests.** `test_simulator.py` has a `# TODO: to fix` comment with commented-out assertion. `test_panel_app.py` has `# assert False` comment. Either complete or remove.

4. **Commented-out test.** `test_vivarium_controller.py` has `test_is_connected_verify_detects_no_server` entirely commented out.

5. **Pre-existing failure.** `test_reproduction` (in `test_components.py`) has a known failure unrelated to spawn.

### Medium Severity

6. **Hardcoded step counts.** `NUM_STEPS` defined as a global in 4 test files (4, 5, 6, 10). Not a bug but inconsistent.

7. **NaN workaround in `test_environments.py`.** Lines 27-29 revert state on NaN detection — indicates potential numerical instability in physics, not a test issue.

8. **Test files don't mirror source structure.** No consistent mapping between test file names and source packages. `test_edu_sessions.py` tests controllers but isn't named accordingly.

### Low Severity

9. **No test markers.** Tests aren't marked as `@pytest.mark.slow` for integration tests or `@pytest.mark.unit` for fast tests. Can't easily run subsets.

10. **Fixture dependency complexity.** The factory-of-factories pattern in conftest.py is powerful but hard to follow. No documentation of fixture relationships.

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

## Structural Observations

### What Works Well
- conftest.py fixture architecture is sophisticated — supports both fast in-process and slow subprocess testing
- `test_edu_sessions.py` provides excellent integration coverage of the student-facing API
- Component factory fixtures compose cleanly via the chain pattern
- Session-level cleanup prevents test pollution

### What Needs Improvement
- **Test organization**: Files don't follow a clear package-mirroring convention. No subdirectories.
- **Fixture documentation**: The factory pattern is powerful but has no docstrings or comments explaining relationships
- **Missing markers**: No way to run fast vs slow tests selectively
- **Coverage gaps**: Core JAX components (physics, sensing, entities) have zero direct tests — only exercised indirectly through integration tests
- **No notebook testing**: No mechanism to verify notebooks run correctly against code changes

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Fix no-op tests: add assertions to `test_instantiate()`, `test_braitenberg()` |
| High | Fix or remove incomplete tests (TODO comments, commented-out code) |
| High | Rename duplicate `test_remote()` functions |
| High | Add test markers (`@pytest.mark.slow`, `@pytest.mark.unit`) |
| Medium | Add direct tests for physics components (collision, friction, reset) |
| Medium | Add direct tests for Braitenberg sensorimotor logic |
| Medium | Document conftest.py fixture relationships |
| Medium | Consider mirroring source structure with test subdirectories |
| Low | Standardize NUM_STEPS across test files |
| Low | Add docstrings to complex test functions |

## Structural Questions (resolved in Phase 1 Step 2)

1. **Should tests mirror source package structure?** Resolved: **keep flat for now.** With ~17 test files, the flat structure is manageable. Reorganizing into subdirectories would complicate conftest.py fixture sharing (session-scoped fixtures, factory chains). Revisit only if test count grows significantly.

2. **Notebook testing strategy.** Deferred to Phase 3. Options remain: (a) reference-solution notebooks tested in CI, (b) metadata-marked hole cells skipped during testing, (c) test only non-session notebooks. Decision depends on Phase 3 documentation scope.

3. **Integration test speed.** Resolved: **keep both modes.** In-process gRPC tests (fast) cover serialization and API correctness. Subprocess tests (slow) cover process lifecycle — a distinct concern. Both add value. Consider adding `@pytest.mark.slow` markers to allow running fast tests only.
