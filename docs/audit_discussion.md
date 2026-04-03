# Audit Discussion Log

_Working document for Phase 1 Step 3 — planning discussion._
_Tracks what has been discussed, agreed, and what remains._
_The numbering of items should be preserved_
_Always ask before starting to discuss a new item in "Remaining to Discuss" or before moving an item to "Discussed & Agreed"_

---

## Discussed & Agreed

### Layer 1 — Architectural / Structural Decisions

#### §1.1 Component package location

_Sources: global CLAUDE.md §Cross-Package Issues #1, environment/CLAUDE.md §Structural Questions, controllers/CLAUDE.md §Known Issues #3, interface/CLAUDE.md, conf/CLAUDE.md §Config–Code Structure Mismatch_

**Decision: Move `vivarium/environment/components/` → `vivarium/components/` (top-level package).**

- Entire directory moves, including base `Component` and `EntityComponent` classes — no splitting.
- Dependency direction becomes clean: `environment/` imports from `components/`, `components/` imports from `controllers/`/`interface/`.
- Re-export shims in `controllers/components/` and `interface/components/` become unnecessary — delete them.

**Workflow:**
1. User moves the directory and uses VSCode refactoring for Python imports
2. Claude updates all `_target_` strings in YAML configs after the commit
3. Verify re-export shims are no longer needed and remove them
4. Run tests to confirm nothing broke

**Corresponding tasks in PLAN.md:** P2.10

#### §1.2 Config–code structure mismatch

_Sources: conf/CLAUDE.md §Config–Code Structure Mismatch, global CLAUDE.md §from_config Pattern, global CLAUDE.md §Cross-Package Issues #5_

**Decision: Defer structural decision to §3.1 (audience journeys).**

- The three-concern pattern (constructor args + `client:` metadata + template directives in one YAML node) is **intentional** — keeps all parameters of a component visible in one place (e.g., entity diameter and color together).
- `get_kwargs()` filtering and `from_config` preprocessing are small prices for config readability.
- At minimum, this is a **Phase 3 documentation task**: document the three-concern convention, `from_config` lifecycle, and template directives.
- May escalate to a **Phase 2 refactoring task** if §3.1 reveals that the pattern is too hard to document clearly or impairs usability for any audience. The audience journey discussion will be the litmus test.
- This might reveal important architectural design flaws that will require significant effort to be solved, so the decision will have to be taken carefully.

**§3.1 ordering:** When we reach §3.1 in this discussion, we do a **lightweight sketch** of each audience journey — just enough to identify where config/init creates friction. This informs the refactor-or-document decision for §1.2 without writing full documentation prematurely. The actual documentation is written in Phase 3 on the stabilized codebase.

**Note from CMF, the developer:** We could try to write a full scene instantiation using only the main `__init__()` constructors for all classes, without any `from_config()` constructor (relates to the previous point, as this might be considered as the Research journey, but with controllers and interfaces in addition). If this is possible, using a more standard hydra initialize pattern when using configs might be a viable option. Potential prompt for Claude: *Write a script/notebook that instantiates the scene in `braitenberg.yaml` using only the main `__init__()` constructors, i.e. without referring to the yaml config. We must be able to start the server and interface and connect a jupyter notebook without using any yaml config file.*

**Update (§3.1.1 discussion):** The `__init__`-only experiment has been completed — see `scripts/dev/config_free_braitenberg.py`. It works with the current codebase. The refactor-or-document decision is: **investigate config modernization** (custom resolvers, structural alignment between configs and `__init__` signatures) via a focused spike on `braitenberg.yaml` before finalizing Phase 2 tasks. See §3.1.1 for the full plan, including prerequisites and timing.

**Corresponding tasks in PLAN.md:** P2.16

#### §1.3 Streaming and real-time update system

_Sources: interface/CLAUDE.md §Known Issues #2-3, simulator/CLAUDE.md §Structural Questions #1 and §Known Issues #10, global CLAUDE.md §Cross-Package Issues #8, conf/CLAUDE.md (use_streaming), scripts/CLAUDE.md (benchmark_streaming_real.py)_

**Decision: Remove bidirectional streaming. Keep one-way streaming, make it always-on in Panel UI, clean up dead code.**

**Investigation findings:**
- Panel UI uses one-way state streaming (`StreamState` RPC, enabled by `use_streaming: true`) — the streaming thread updates `client.state` asynchronously, while `apply_changes()` sends changes via fire-and-forget `SetChanges`.
- Bidirectional stepping is fully implemented but dormant — no active code path uses it.
- `_pending_state_update` flag was intended to make UI event-driven but the check was never wired into `update_plot_cb()`.
- The `is_streaming` conditional in `apply_changes()` must stay — it correctly distinguishes Panel UI (streaming, fire-and-forget changes) from notebooks (no streaming, `SetChangesReturnsState`).

**Remove (bidirectional streaming — dormant):**
- `bidirectional_step_sync()` and `bidirectional_step_generator()` in `simulator_client.py`
- `BidirectionalStep` RPC handler in `simulator_server.py` + proto definition (regenerate)
- `use_streaming` parameter in `VivariumController.start_controller_thread()`
- `scripts/dev/benchmark_streaming_real.py`

**Remove (dead UI code):**
- `streaming_toggle` widget + `streaming_toggle_cb` callback
- `use_streaming` config key in `base_interface.yaml` (streaming always on in Panel UI)

**Fix:**
- Wire `_pending_state_update` flag into `update_plot_cb()` — skip repaint when no new state has arrived

**Keep as-is:**
- `StreamState` RPC, `start_state_stream()`/`stop_state_stream()`, `_start_streaming()`/`_stop_streaming()`
- `is_streaming` conditional in `apply_changes()` / `set_changes()`

**Corresponding tasks in PLAN.md:** P2.12

#### §1.4a `utils/` reorganization and `vivarium/runtime/` package

_Sources: global CLAUDE.md §Cross-Package Issues #2, utils/CLAUDE.md §Known Issues #1 and §Refactoring Opportunities, scripts/CLAUDE.md §Known Issues #3_

**Decision: Create a new top-level `vivarium/runtime/` package. Split `handle_server_interface.py` into modules, move it there along with `runtime.py` (renamed `paths.py`) and `updater.py`. Clean up `converters.py`. Keep `utils/` as a lean shared foundation package.**

`handle_server_interface.py` (881 LOC) mixes five concerns (process management, server lifecycle, interface lifecycle, Jupyter, ngrok) and `runtime.py` provides the underlying path/command logic they all depend on. Together with `updater.py` (update checking/download), these form a cohesive "how vivarium runs as a system" package — distinct from `utils/` which is cross-cutting domain code (config loading, change protocol, data helpers).

**New `vivarium/runtime/` package:**
```
vivarium/runtime/
  __init__.py       # re-exports public API
  paths.py          # is_frozen, get_app_root, get_*_command (was utils/runtime.py)
  updater.py        # update checking, download, post-update merge (was utils/updater.py)
  _process.py       # PID lookup, kill, terminate
  _server.py        # gRPC server start/stop/wait
  _interface.py     # Panel interface start/stop
  _jupyter.py       # Jupyter start/stop/check, port management
  _ngrok.py         # tunnel creation, colab detection
```

**Resulting `vivarium/utils/` — kept as a "shared foundation" package** (cross-cutting domain code used by multiple packages, no internal vivarium deps beyond utils itself):
```
vivarium/utils/
  __init__.py
  scene_configs.py      # Hydra config loading, scene enumeration, template expansion
  dataclass_wrapper.py  # Remote proxy, change protocol, dataclass updates
  timer.py              # SleepTimer
  jax_utils.py          # JAX-MD dataclass detection
```

**Clean up `converters.py`:** Most functions are dead code (`upper_camel_to_snake`, `snake_to_upper_camel`, `class_import_path`, `import_class` — none used anywhere). Only `access_nested_fields` is live (used by `environment.py`). Move it to an appropriate location (e.g. inline in `environment.py` or keep in utils under a better name), then delete `converters.py`.

**Workflow:**
1. Create `vivarium/runtime/` package with `__init__.py`
2. Move and rename `utils/runtime.py` → `runtime/paths.py`
3. Move `utils/updater.py` → `runtime/updater.py`
4. Split `utils/handle_server_interface.py` into the 5 internal modules
5. Update `runtime/__init__.py` to re-export public API
6. Update all imports across the codebase
7. Update `utils/__init__.py` (stop re-exporting moved functions, or keep shims temporarily)
8. Run tests

**Note from the developer:**
It might be more efficient to leverage the refactoring functionalities of VSCode to move modules and functions. VSCode should adapt the imports automatically. This can only be done by human though (I think).

**Corresponding tasks in PLAN.md:** P2.13

#### §1.4b `panel_app.py` monolith split

_Sources: global CLAUDE.md §Cross-Package Issues #2, global CLAUDE.md §Audience Journey Readiness #3, interface/CLAUDE.md §Known Issues #1 and §Refactoring Opportunities and §Structural Questions #3_

**Decision: Extract `UpdateManager` and `JupyterManager` into separate modules. Rename `WindowManager` → `PanelApp`. Keep `panel_app.py` filename. Don't split the remaining ~760 LOC core further.**

- **`UpdateManager`** (~340 LOC) → `update_manager.py` — self-contained: owns its widgets, background thread, calls `updater.py` functions. No dependency on the main app beyond being placed in the layout.
- **`JupyterManager`** (~310 LOC) → `jupyter_manager.py` — needs `scene_config` (notebook path/port) and `controller` (notebooks dir) passed as constructor args.
- **`PanelApp`** (~760 LOC) stays in `panel_app.py` — scene selection, Bokeh plot, controls, streaming, component tabs, callbacks. These share state heavily (`self.controller`, `self.interfaces`, `self.main_area`, widgets) — splitting further would require an awkward shared-state object.

**Naming rationale:** `PanelApp` rather than `VivariumApp` or `VivariumInterface` — avoids stutter with `vivarium.interface.VivariumInterface`, and distinguishes from component `Interface` subclasses (`BraitenbergInterface`, etc.) which are parts that plug into the app.

**Resulting structure:**
```
vivarium/interface/
  panel_app.py          # PanelApp (~760 LOC) — core simulation UI
  update_manager.py     # UpdateManager (~340 LOC) — update checking/download
  jupyter_manager.py    # JupyterManager (~310 LOC) — Jupyter lifecycle/UI
  parameterized.py      # (unchanged)
  utils.py              # (unchanged)
```

**Corresponding tasks in PLAN.md:** P2.14

#### §1.5 Dynamic dataclass patterns

_Sources: global CLAUDE.md §Cross-Package Issues #3, simulator/CLAUDE.md §Known Issues #4 #5, interface/CLAUDE.md §Known Issues #5_

**Decision: Accept dynamic `ControllerParameters` as-is. Fix ghost attributes by eliminating `update_from_dataclass()` — always access through `self.controller_parameters.simulator`. Build a default `controller_parameters` in `__init__` when none is provided.**

**Context:** `Simulator.from_config()` builds `ControllerParameters` at runtime via `create_dataclass_from_dict()`. The dynamic dataclass pattern works well for component fields (entity types access them dynamically via `getattr(cp, entity_type)`). The real problem is narrower: `update_from_dataclass()` copies `run_from`, `simulation_running`, and `freq` from `controller_parameters.simulator` onto the `Simulator` instance as undeclared ghost attributes.

**Fix:**
1. **Remove the `update_from_dataclass()` call** in `set_changes()` — eliminates ghost attributes entirely
2. **Replace direct `self.X` accesses** with `self.controller_parameters.simulator.X` for `run_from`, `simulation_running` (3 locations in `set_changes()` and `run()`)
3. **Fix `freq` aliasing** — `to_config()` uses `self.freq` but `__init__` stores `self._freq`. Change to access `self.controller_parameters.simulator.freq`
4. **Build default `controller_parameters` in `__init__`** when none is provided, so the headless path (`Simulator(env=env, freq=10)`) and the config path work identically:
   ```python
   if controller_parameters is None:
       controller_parameters = create_dataclass_from_dict('ControllerParameters', {
           'simulator': {'freq': freq, 'scene_name': scene_name,
                        'run_from': 'server', 'simulation_running': False,
                        'client_names': []}
       })
   ```
5. **Remove `update_from_dataclass()` function** (top of simulator.py) — no longer needed
6. **Remove `nested_fields_to_access` dead code** (simulator.py:36-43)
7. **Drop misleading `self = ` assignment** in `set_changes()` — `update_dataclass_from_change_list` mutates the Simulator in place via `setattr` (it's not a JAX-MD dataclass), so the return value is the same object. Replace with a plain call + comment:
   ```python
   # Mutates self.state and/or self.controller_parameters in place
   update_dataclass_from_change_list(self, changes)
   ```

**Not changed:**
- `ControllerParameters` stays dynamic — the `getattr(cp, entity_type)` pattern is actually well-suited for extensible component fields
- `StateAndControllerParameters` dual definition — low risk, defer
- Dynamic Param field cleanup in `interface/utils.py` — low risk if no new entity types added

**Corresponding tasks in PLAN.md:** P2.11 (ghost attribute fixes), D.14 (StateAndControllerParameters dual definition), D.15 (dynamic Param field cleanup)

#### §1.6 Script entry points: redundancy and cleanup

_Sources: scripts/CLAUDE.md §Known Issues #3 #5, utils/CLAUDE.md §Known Issues #2_

**Decision: Remove `run_vivarium.py` entirely.**

- `run_interface.py` already handles the full workflow: scene selection UI → server startup → interface. It has proper cleanup (signal handlers, atexit, process killing).
- `run_vivarium.py` is **not** in the PyInstaller `.spec` file — the spec builds `vivarium-server`, `vivarium-interface`, and `vivarium-jupyter` from their respective scripts.
- Not used programmatically — `handle_server_interface.py` spawns `run_server.py` and `run_interface.py` directly.
- No test coverage. The 4 layers of PyInstaller guards (spawn counter, env vars, child process detection) are dead code since it's never bundled.
- Removes the inconsistent scene argument style issue (positional arg vs Hydra override).

**Corresponding tasks in PLAN.md:** P2.03

### Layer 2 — Per-Package Cleanup

#### §2.1 Dead code removal

_Sources: global CLAUDE.md §Dead Code to Remove, per-package CLAUDE.md files_

**Decision: Remove the following dead code across all packages.**

**Files to remove:**
- `vivarium/environment/components/eco_evo/component.py` — empty file (0 bytes)
- `scripts/run_vivarium.py` — redundant combined launcher (decided in §1.6)
- `notebooks/sessions/session_5_logging copy.ipynb` — older draft with defunct `logger.plot()` content

**Files to fix:**
- `scripts/print_config.py` — test if it still works, fix if needed
- `scripts/profiling.py` — adapt to current API (uses non-existent `SceneConfiguration` import)

**Code to remove:**
- `nested_fields_to_access` dict (simulator.py:36-43) + commented-out decorator reference (simulator_client.py:24) + unused variable (environment.py:71-73). Keep `access_nested_fields` function and `@access_nested_fields(...)` decorator on `Environment` — actively used.
- `SetState` RPC handler (simulator_server.py:112-118) + proto definition — calls non-existent `simulator.set_state()`. Regenerate proto.
- Recording feature (`record`, `start_recording`, `stop_recording`, `save_records`, `load`) in simulator.py. Replacement mechanism tracked in §3.3.
- `self.notebook_mode` parameter and assignment (panel_app.py)
- Commented-out config_update logic (panel_app.py:993-994, 999)
- `kill_session()` in controllers/utils.py
- `set_nested_attr` from controllers `__init__.py` exports (no consumer)

**Corresponding tasks in PLAN.md:** P2.06

#### §2.1a Rigid body removal

_Sources: global CLAUDE.md §Cross-Package Issues #4, environment/CLAUDE.md §Structural Questions #1, environment/CLAUDE.md §Known Issues (rigid_body_to_point_particle)_

**Decision: Remove all rigid body support entirely.** No scene uses rigid bodies, no tests exercise the rigid body paths, and the conditional branching adds complexity throughout the codebase.

**Investigation completed — full inventory of rigid body code paths:**

**State system — `state.py`:**
- `from jax_md.rigid_body import RigidBody` import (line 4)
- `to_rigid_body_state()` function (lines 8-21)
- `is_rigid_body()` method on BaseEntityState (lines 50-51)
- `unified_*` accessors in `BaseEntityState.__getattr__` (lines 56-73) — branch on `is_rigid_body()` to extract `.center`/`.orientation`
- RigidBody wrapping branch in `BaseState.__getattr__` (lines 138-146)

**Physics components — 5 files:**
- `components/utils.py`: `to_rigid_body()`, `@handle_rigid_body` decorator, RigidBody path in `sum_forces()` (lines 6, 16-31)
- `reset/component.py`: `if is_rigid_body()` → zero forces as RigidBody (lines 4, 11-16)
- `collision/component.py`: `@handle_rigid_body` decorator + `if is_rigid_body()` force set (lines 8, 32, 116-122)
- `friction/component.py`: `@handle_rigid_body` decorator + `if is_rigid_body()` force set (lines 4, 7, 38-44)
- `step/component.py`: `mask_momentum()` branches on `is_rigid_body()` (lines 3, 19-33)

**Braitenberg sensorimotor — `sensorimotor.py`:**
- `motor_force()` and `sum_force_to_entities()` branch on `is_rigid_body()` for separate rotational dynamics (lines 6, 109-138)

**Entity controllers — `entities/controller.py`:**
- `create_property()` function (lines 20-54) — entire function exists to provide RigidBody-style `*_center`/`*_orientation` access. Remove entirely.
- 8 class-level properties on `EntityWrapper` (lines 64-72): `position_center`, `momentum_center`, `force_center`, `mass_center`, `position_orientation`, `momentum_orientation`, `force_orientation`, `mass_orientation`. Remove all.
- `self._is_rigid_body` flag in `__init__` (line 78). Remove.
- The 8 property names appended to `_entity_fields` list (lines 82-87). Remove.
- `_center`/`_orientation` suffix handling in `_setitem()` (lines 100-103). Remove branch.
- `is_rigid_body` check in `EntityController.__getattr__` (lines 158-159). Remove.
- `split()` helper (line 16): remove `+ '_center' if suffix == 'position' else suffix` — a RigidBody artifact that maps `left_position` → `position_center`.

**Deprecated utility — `environment/utils.py`:**
- `rigid_body_to_point_particle()` (lines 49-91) — marked deprecated, never called

**gRPC layer:**
- `RigidBody` message in `simulator.proto` (lines 151-154) — defined but never used. Remove + regenerate proto.
- `RigidBody` import in `grpc_server/converters.py` (line 5) — unused

**Consumers of RigidBody-style accessors (`position_center`, `position_orientation`, etc.):**
- `render.py`: `state.entity_state.position_center` (lines 17, 38), `state.entity_state.position_orientation` (line 41) → replace with `.position` and `.orientation`
- `test_components.py:113`: `state.entity_state.position_center` → `.position`
- `test_param.py:22,26`: `entity.position_center` → `.position`; line 36: `entity.position_orientation` → `.orientation`
- `test_vivarium_controller.py:31`: `ag.position_center` → `.position`
- `test_grpc.py:28`: `state.entity_state.position_center` → `.position`

**Tests (RigidBody-specific):**
- `test_dataclass_api.py`: `get_rigid_body_state` fixture, RigidBody parametrization, two `test_remote()` tests exercising RigidBody (lines 5, 19-22, 53, 87-114)
- `test_environments.py`: `assert not state.entity_state.is_rigid_body()` (line 33)

**`unified_*` accessor replacement (Option B — replace with direct field access):**

The `unified_*` accessors exist solely for the RigidBody abstraction. For point particles they are trivial pass-throughs (`unified_position` → `self.position`). After rigid body removal, replace all 8 call sites with direct field access and remove the `unified_*` `__getattr__` logic:

| Accessor | File | Line(s) |
|----------|------|---------|
| `unified_position` | `environment.py` | 161, 213 |
| `unified_position` | `sensorimotor.py` | 106 |
| `unified_orientation` | `sensorimotor.py` | 88 |
| `unified_momentum` | `sensorimotor.py` | 99 |
| `unified_mass` | `sensorimotor.py` | 100 |
| `unified_momentum` | `step/component.py` | 18 |
| `unified_force` | `step/component.py` | 30 |
| `unified_momentum`, `unified_mass` | `friction/component.py` | 15 |

**Keep untouched:**
- `scripts/patch_jax_md.py` — patches jax_md's own `rigid_body.py` for JAX compatibility, unrelated to vivarium's RigidBody usage
- Sandbox notebook — not in release scope
- Outdated notebooks referencing RigidBody — handled separately in §3.2

**Corresponding tasks in PLAN.md:** P2.07

#### §2.2 Bugs to fix

_Sources: environment/CLAUDE.md §Known Issues #2 #4, interface/CLAUDE.md §Known Issues #4 #6, global CLAUDE.md §Patterns to Fix #6 #7, simulator/CLAUDE.md §Known Issues #6 #7, utils/CLAUDE.md §Known Issues #2, conf/CLAUDE.md §Known Issues #2 #3 #5_

**Decision: Fix 10 bugs, skip 1.**

**Rendering (`render.py`) — do after §2.1a:**
- Remove double `[exists]` indexing on diameter (line 18) and orientation (lines 41-42)
- Fix `plt.xlim` → `plt.ylim` (line 67)
- (Accessor renames `position_center` → `position` etc. covered by §2.1a)

**Typos and code smells:**
- Rename `udpate_other_interfaces` → `update_other_interfaces` in `environment/components/interface.py:65` and `interface/panel_app.py:645`
- `except:` → `except Exception:` in `utils/handle_server_interface.py:469`
- `lg.error(...)` → `lg.exception(...)` in `panel_app.py:852` (`_start_server_cb`)

**Config:**
- Add `- _self_` at end of defaults list in `braitenberg.yaml` and `particle_lenia.yaml`
- Remove 60+ lines of commented-out code in `demo.yaml`

**Simulator — do during/after §1.4a:**
- Replace hardcoded `'../../conf/scene/simulator'` in `to_config()` with `runtime.get_config_dir()`
- Add gRPC error handling decorator: catch `Exception`, `lg.exception()`, set `INTERNAL` status + details

**Skipped:**
- `start_server_and_interface()` missing function — only used by outdated notebooks being rewritten/removed in §3.2

**Corresponding tasks in PLAN.md:** P2.09

#### §2.3 Test suite cleanup and gaps

_Sources: tests/CLAUDE.md §Known Issues #1-10 and §Coverage Analysis and §Refactoring Opportunities, environment/CLAUDE.md §Test Coverage, controllers/CLAUDE.md §Test Coverage, simulator/CLAUDE.md §Test Coverage, interface/CLAUDE.md §Test Coverage, utils/CLAUDE.md §Test Coverage, conf/CLAUDE.md §Test Coverage_

**Testing strategy for the release:** Make the existing suite honest and developer-friendly. Defer writing new test files for untested modules — indirect coverage from integration tests is sufficient for a "clean and usable" release.

**Test quality fixes (Phase 2):**

1. **No-op tests:** Add minimal assertions to `test_instantiate()` (assert factory count > 0, expected names present) and `test_braitenberg()` (assert no NaN, state changed after step).

2. **Duplicate `test_remote()` names in `test_dataclass_api.py`:** Rename to `test_remote_fetch_and_update()` and `test_remote_apply()`. First test is currently shadowed and never runs. Both switch to point particle fixture as part of §2.1a.

3. **Commented-out `test_is_connected_verify_detects_no_server`:** Uncomment and fix. Valuable edge-case test (stale connection detection). Monkeypatch approach keeps it fast.

4. **Incomplete tests with TODOs:**
   - `test_simulator.py:38` (`# TODO: to fix`) — leave the TODO, revisit after §1.5. The controller_parameters round-trip may work once §1.5 lands.
   - `test_panel_app.py:22` (`# assert False`) — remove the comment. The test is functional (has real assertions on lines 12-13).

5. **`test_reproduction` pre-existing failure:** Investigate and fix in Phase 2 (not deferred to Phase 4). It used to pass — likely a regression with bounded debugging effort. Run in isolation, trace the error, fix.

6. **NaN workaround in `test_environments.py`:** Remove the silent revert logic (lines 27-29). Replace with `assert not jnp.isnan(state.entity_state.position).any()`. If specific scenes produce NaN, mark those with `@pytest.mark.xfail` rather than silently recovering.

11. **Student-facing API coverage (expanded — see also §3.2):** Exhaustively list all features demonstrated in sessions 1-4 + miniproject, then write tests covering all of them in `test_edu_sessions.py`. This includes (non-exhaustive): Logger (`agent.logger.add/get`), routines (`attach_routine/detach_routine`), multiple behaviors with weights, entity property access (energy, exists, position read/write), selective sensing, consumption/spawning settings, scene customization via Hydra overrides. Do **early in Phase 2** — before major refactors (§1.1, §1.4a, §1.2 experiment) — these tests serve as a safety net against regressions. The feature inventory also serves as the basis for the main tutorial (§3.1.2).

**Coverage gaps — all deferred post-release:**

- Physics components (collision, friction, reset), sensorimotor, consumption, walls, ProximityMap, Particle Lenia — all exercised indirectly via integration tests (`test_environments.py`, `test_edu_sessions.py`). Direct unit tests would require domain knowledge (expected numerical outcomes). Defer.
- Controllers internals (Logger, RoutineHandler, BehaviorHandler) — well-exercised by `test_edu_sessions.py` (31 tests). Defer.
- Simulator (SimulatorController, streaming rate limiting) — thin wrappers, exercised by integration tests. Defer.
- Interface — smoke test only (current `test_panel_app.py`). After §1.4b split, add unit tests for `UpdateManager` and `JupyterManager` (self-contained modules). Don't invest in deeper `PanelApp` testing — effort-to-value ratio is poor given Bokeh coupling. Manual testing in Phase 5 is the real quality gate.
- Utils (scene_configs partial, handle_server_interface, runtime command builders, converters, timer, jax_utils) — `scene_configs` and `runtime` already have good coverage. `handle_server_interface` is being split into `vivarium/runtime/` (§1.4a) — test the new modules as part of that task. `converters.py` is mostly dead code being removed. `timer.py` and `jax_utils.py` are trivial. Defer.

**Coverage gap — do now (trivial):**

- Add 3 untested scene configs (`demo`, `excretion`, `boyds`) to `test_environments.py` parametrize list. One line of effort.

**Structural improvements:**

7. **Test markers:** Add `@pytest.mark.slow` to integration test files (~7 files). Register the marker in pytest config. Don't add `@pytest.mark.unit` — unmarked tests are implicitly fast.

8. **Fixture documentation:** Add docstrings to each fixture in `conftest.py`. One-liner when sufficient, longer when required.

9. **`NUM_STEPS` inconsistency:** Leave as-is. Different values serve different purposes (smoke test vs behavioral simulation). Not a real inconsistency.

10. **Test directory restructuring:** Restructure `tests/` to mirror source package structure. Do as one of the first Phase 2 tasks, before §1.1 (component move), so new tests land in the right place.

**Target structure:**
```
tests/
  conftest.py                    # root: shared fixtures (scene_config, server lifecycle, cleanup)
  __init__.py
  environment/
    __init__.py
    conftest.py                  # component fixture chain (step → braitenberg → ...)
    test_environment.py          # was test_environments.py
    test_state.py
    test_components.py
    test_multi_spawn.py
  simulator/
    __init__.py
    test_simulator.py
    test_grpc.py
  controllers/
    __init__.py
    test_vivarium_controller.py
    test_edu_sessions.py
  interface/
    __init__.py
    test_panel_app.py
    test_param.py
  utils/
    __init__.py
    test_runtime.py
    test_version.py
    test_scene_config.py
    test_updater.py              # was test_update_check.py
    test_dataclass_wrapper.py    # was test_dataclass_api.py
  scripts/
    __init__.py
    test_start_stop_scripts.py
```

**Conftest split:** Root conftest keeps widely-used fixtures (~300 LOC): session cleanup, `scene_config`, `simulator_from_config`, `grpc_server`/`grpc_client`, `vivarium_controller*`, server subprocess fixtures. Component fixture chain (~120 LOC: step, braitenberg, spawn, proximity_map, consumption, energy, reproduction, environment, environment_and_state) moves to `environment/conftest.py`.

**Known issue to handle:** `test_multi_spawn.py` has `from conftest import remove_duplicates` — inline the function (5 lines) into the test file.

**Prerequisite:** Fix `test_reproduction` first (item 5 above), so the baseline is all tests passing.

**Verification:** Run full `pytest` before restructuring to confirm all green. Then `pytest --collect-only` before and after — same test count, no collection errors. Then full `pytest` — still all green.

**Deferred (optional, post-release):**
- Stub files for untested modules (makes gaps visible in file tree)
- Directory-level markers (use per-file markers first)

**Corresponding tasks in PLAN.md:** P2.01 (item 5 — test_reproduction fix), P2.02 (item 11 — exhaustive feature tests), P2.04 (items 1-4, 6-8 — test quality fixes), P2.05 (item 10 — test directory restructuring), D.16 (coverage gaps deferred), D.17 (structural improvements deferred)

### Layer 3 — Documentation & Feature Scope

#### §3.1 Audience readiness and documentation plan

_Sources: global CLAUDE.md §Audience Journey Readiness #1-3, PLAN.md §Phase 3, notebooks/CLAUDE.md §Known Issues #2-3 and §Structural Questions #1 #3 #4, environment/CLAUDE.md §Structural Questions #2 (render.py as headless output), interface/CLAUDE.md §Structural Questions #1 (Bokeh headless). Also resolves §1.2 (config–code structure)._

Preliminary note: We should decide how to name these journeys. Currently they are called Researcher, CS student and Web interface journeys, but this might not be the best naming.

##### 3.1.1 Researcher journey (headless JAX) — AGREED

**Context:** Researchers want to run large-scale headless simulations in JAX. Two API levels exist: `Environment` (pure JAX step loop) and `Simulator` (higher-level wrapper). Both work, but there was zero documentation and no working notebooks. Server-side notebooks all use obsolete import paths.

**Key finding — config-free instantiation works today:** A validation script (`scripts/dev/config_free_braitenberg.py`) was written and successfully builds the full braitenberg scene (components, Environment, Simulator) using only `__init__()` constructors — no YAML, no Hydra, no `from_config()`. This resolves the §1.2 `__init__`-only experiment: the `__init__` signatures are clean enough for programmatic use.

**Architectural gap — client-config reload:** When a researcher builds a Simulator programmatically and starts a gRPC server, clients (`VivariumController`) currently reload the scene config from disk (`load_scene_config(scene_name)`). If no YAML exists on disk, the client can't discover which controller classes to instantiate. The client needs exactly two things from the config: (a) which controller class to use for each component (`controller_cls` strings), and (b) controller kwargs (already transmitted as `controller_parameters`).

**Decision — transmit `controller_cls` via `controller_parameters`:**
- Store `controller_cls` (and `interface_cls`) at the component level in `controller_parameters` — alongside but separate from `controller_kwargs` (which are constructor arguments). Example: `controller_parameters.agents.controller_cls`.
- In `Simulator.from_config()`, extract `controller_cls` from the `client:` block when building `controller_parameters` (currently only `controller_kwargs` is extracted).
- If the user want to launch a server from an existing `Simulator` instance, it first need to fill relevant fields in the `controller_parameters` attribute of the `Simulator`.
- Client reads `controller_parameters.<component>.controller_cls` to discover which class to instantiate — no config file needed.
- No new proto message or RPC required — `controller_parameters` is already transmitted via `GetControllerParameters`.

**Decision — §1.2 config experiment before finalizing Phase 2:**
Rather than just documenting the config conventions, investigate whether configs can be modernized to better align with `__init__` signatures. Specifically:
- Try restructuring `braitenberg.yaml` so that config structure mirrors `__init__` signatures more closely.
- Investigate OmegaConf custom resolvers (see https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html) to replace Python-side template directives (`_all_values_`, `_range_`, `_random_`; `by_indices` is harder). Currently only a `range` resolver exists in `scene_configs.py`.
- Investigate whether Hydra Structured Configs (https://hydra.cc/docs/tutorials/structured_config/intro/) can help.
- It's OK to keep `get_class().from_config()` rather than switching to `hydra.utils.instantiate()` if it provides more flexibility — the goal is structural alignment between configs and `__init__`, not adopting a specific Hydra API.
- Scope: single scene (`braitenberg.yaml`) as a spike. Outcome informs whether config restructuring goes into Phase 2 (if viable and bounded) or stays as a documented future direction.

**Timing — prerequisites before the config experiment:**
1. Fix `test_reproduction` → all tests green (§2.3 item 5)
2. Test directory restructuring (§2.3 item 10)
3. §1.1 Component package move (changes all `_target_` strings — do once, not twice)
4. §2.1 Dead code removal + §2.1a Rigid body removal (simplifies `__init__` signatures and configs)
5. §1.5 Dynamic dataclass fixes (changes how `controller_parameters` is built — directly relevant)
6. **§1.2 Config experiment** — restructure `braitenberg.yaml`, test resolvers, assess effort

**Phase 3 documentation (on stabilized codebase):**
- Rewrite a new notebook as a researcher tutorial covering: config loading, Environment vs Simulator, stepping, data extraction, visualization with `render.py`.
- Archive remaining server-side notebooks.
- Write a separate tutorial for custom scene creation in YAML (distinct from the researcher tutorial).
- `scripts/dev/config_free_braitenberg.py` serves as a reference for the `__init__`-only path.
- Explain how to install with extra for GPU support (see cuda-related extras in `setup.py`). Either in the main README or in a research-oriented README.

**Corresponding tasks in PLAN.md:** P2.15 (transmit controller_cls), P2.16 (config experiment), P3.01 (researcher tutorial), P3.02 (YAML scene creation tutorial), P3.03 (archive server-side notebooks)

##### 3.1.2 CS student journey (programmatic control) — AGREED

**Context:** Sessions 1-4 are excellent and well-tested (31 tests in `test_edu_sessions.py`). `miniproject_template.ipynb` and `reactive_rl.ipynb` are functional. Sessions target interdisciplinary Master students with limited programming background.

**Decisions:**

**Delete:**
- Session 5 (`session_5_logging.ipynb`) — content integrated into miniproject notebook.
- Session 6 (`session_6_bonus.ipynb`) — uses deprecated API (`kill_session()`), content covered by miniproject.
- `quickstart_tutorial.ipynb` — outdated (uses `NotebookController`). Session 1 already serves as the entry point for the sessions audience, and the new main tutorial (below) will serve CS students.

**Keep as-is:**
- Sessions 1-4 + miniproject + `reactive_rl.ipynb` — for the interdisciplinary Master audience. These coexist permanently with the main tutorial.

**Write (Phase 3) — new main tutorial:**
- Target audience: CS students comfortable with standard Python programming.
- Covers all key concepts from sessions 1-4 and the miniproject in compact form — behaviors as functions, selective sensing, routines, logging, custom configs, etc. — without the verbose explanations needed for non-programmers.
- Starting basis: the miniproject notebook, which assumes prior session knowledge. The main tutorial fills in those assumptions but in a concise style.
- **Phase 3 prerequisite:** Before drafting the structure, exhaustively list all features to document. The miniproject's Recap section provides a starting point, plus features introduced in the miniproject itself.

**No validation script needed** — existing sessions + `test_edu_sessions.py` already validate this journey.

**Deferred post-release:**
- Standalone API reference documentation.

**Corresponding tasks in PLAN.md:** P2.03 (notebook deletions), P3.04 (main CS tutorial), D.18 (standalone API reference deferred)

##### 3.1.3 Web interface journey (non-programmers) — AGREED

**Context:** The Panel UI works (scene selection, Bokeh visualization, drag-drop, config tabs, start/stop). Only documentation is `web_interface_tutorial.md` (81 lines, current). `sessions/README.md` covers the PyInstaller binary workflow. The UI will change internally during Phase 2 (§1.4b monolith split, §1.3 streaming cleanup) but user-facing behavior should remain the same. This journey is interactive — students use the UI directly, not code.

**Decisions:**

**Revise `web_interface_tutorial.md` (Phase 3, after §1.4b UI refactoring):**
- The current tutorial covers basics but needs revision to be a proper standalone guide.
- Should demonstrate a concrete use case: setting up a relatively complex simulation using only the interface (no code).

**Phase 3 prerequisite — discuss tutorial structure before writing:**
- Agree on what should be explained (all UI features: scene selection, entity interaction, config tabs, Jupyter embedding, etc.).
- Decide on structure: explain all parts of the interface first then demonstrate the use case, vs. use it as a running example throughout the tutorial.
- Choose the concrete use case.

**Manual validation in Phase 5:**
- Walk through the full journey: launch interface → scene selection → agent interaction → config changes → Jupyter embedding. Document rough edges found.

**No validation script** — this journey is interactive by nature.

**Corresponding tasks in PLAN.md:** P3.05

##### 3.1.4 Developer journey (extending the code) — AGREED

**Context:** No documentation exists for developers who want to extend vivarium. A "how to add a component" guide is critical for an incoming Master's student (mentioned in PLAN.md Phase 3).

**Decisions:**

**Write a notebook** (e.g. `notebooks/tutorials/adding_a_component.ipynb`) in Phase 3, covering:
1. Implementing a new component from scratch — the three-file pattern (`component.py`, `controller.py`, `interface.py`) and the component lifecycle (`update_state_cls` → `init_base_entity` → `init_state_fn` → `get_step_function`).
2. Using it via the `__init__`-only path — instantiate the component, add to Environment, step, verify behavior. This is the primary path presented.
3. Integrating it into the library with YAML configs — as an extra. Covers `_target_` wiring, `client:` block, defaults composition, testing.

**Phase 3 prerequisite — discuss concrete example:**
- Either reimplementing an existing component (simpler, can verify against known behavior) or implementing a new one (more motivating, shows the full creative process).
- Decide before writing.

**Timing:** One of the last Phase 3 tasks — requires stable component architecture (after §1.1 component move, §1.2 config experiment, §2.1a rigid body removal).

**Corresponding tasks in PLAN.md:** P3.06

##### 3.1.5 Cross-cutting documentation decisions — AGREED

**A. Docstring format:**

Codebase uses mixed formats (~150 Google-style, ~172 reST/Sphinx). ~29% coverage overall.

**Decision: Standardize on Google-style docstrings.**
- Most readable as plain text, widely adopted in ML/scientific Python (JAX, TensorFlow, NumPy internals).
- Works with Sphinx via the `napoleon` extension.
- **Mass-convert existing reST docstrings** as a Phase 2 task (or early Phase 3). Use automated tools (`pyment` or `docconvert`) + review.
- **Priority for new docstrings:** Core abstractions first (`Environment`, `Component`, `Simulator`, `VivariumController`), then student-facing APIs.

**Corresponding tasks in PLAN.md (§3.1.5A):** P2.17 (format conversion), P3.10 (new docstrings and type hints)

**B. Documentation website:**

**Decision: Defer post-release.** Tutorials (notebooks) + README + docstrings are sufficient for a "clean and usable" release.

**Corresponding tasks in PLAN.md (§3.1.5B):** D.18

**C. Main README revision:**

**Decision: Revise in Phase 3.** Make the audience journeys explicit. Structure: project description → the four journeys with links to their respective tutorials → installation → development setup.

**Corresponding tasks in PLAN.md (§3.1.5C):** P3.07

**D. Local READMEs and install documentation:**

- **Update** `notebooks/README.md` after notebook changes settle.
- **Delete** `notebooks/tutorials/README.md` (outdated, replaced by actual tutorials).
- **Keep** `notebooks/sessions/README.md` as-is (serves PyInstaller users).
- **Delete** `notebooks/server_side/README.md` (empty, server-side notebooks being archived).
- **PyInstaller binary install:** Move to a standalone document (e.g. `docs/install_binary.md` or a section in the main README). This is the recommended install path for both the web interface journey and the practical sessions. Currently buried in `notebooks/sessions/README.md`.

**E. Existing documentation to fix or relocate:**

- **Google Colab** (`google_colab.ipynb` / `google_colab.md`): Keep, fix hardcoded branch reference and TODOs. Decide on final location in Phase 3.
- **Troubleshooting**: Need to document common issues (e.g. server/client disconnection). Decide in Phase 3 whether to include inline in tutorials or as a dedicated document.

**Corresponding tasks in PLAN.md (§3.1.5D):** P3.08
**Corresponding tasks in PLAN.md (§3.1.5E):** P3.09

**F. Journey naming:**

Currently named by audience (Researcher, CS student, Non-programmer, Developer). Agreed to rename by mode (what you do, not who you are) — avoids assumptions about the user.

| Current name | By activity | By mode |
|-------------|------------|---------|
| Researcher journey | Headless simulation | Programmatic JAX |
| CS student journey | Programmatic control | Notebook control |
| Non-programmer journey | Web interface | Interactive UI |
| Developer journey | Extending the code | Component development |

Additionally, a journey is needed for practical session students (interdisciplinary Master) who use the PyInstaller bundle + sessions 1-4 + miniproject + reactive RL. Two options:
- **Option A — Separate journey ("Practical sessions"):** Distinct entry point, 5th journey.
- **Option B — Branch of "Notebook control":** Two entry paths within the same journey — (a) pip install → main tutorial (CS students), (b) PyInstaller bundle → sessions 1-4 → miniproject → reactive RL (practical sessions).

To decide when we start writing the README/tutorials.

**Timing for renaming:** Decide final journey names at the start of Phase 3, before writing the README and tutorials. Names don't affect code, configs, or tests — no reason to rename earlier.

**Corresponding tasks in PLAN.md (§3.1.5F):** P3.00

#### §3.2 Notebook decisions

_Sources: notebooks/CLAUDE.md §Known Issues #4 #6 #7 and §Structural Questions #2 #3 #4, tests/CLAUDE.md §Structural Questions #2_

Most items already resolved by §3.1 decisions. Remaining items:

**Already decided in §3.1:**
- Tutorials folder: `quickstart_tutorial.ipynb` deleted (§3.1.2), new main tutorial written in Phase 3 (§3.1.2), `adding_a_component.ipynb` written in Phase 3 (§3.1.4).
- Session 6: deleted (§3.1.2).
- Google Colab: keep and fix (§3.1.5E).
- Server-side notebooks: archive, write new researcher tutorial (§3.1.1).

**Notebook testing — decision: write exhaustive feature tests early in Phase 2.**
- Exhaustively list all features demonstrated in sessions 1-4 + miniproject.
- Write tests covering all of them in `test_edu_sessions.py` (expanding §2.3 item 11).
- Do this **before** major refactors (§1.1, §1.4a, §1.2 experiment) — these tests serve as a safety net to catch regressions.
- The feature inventory also serves as the basis for the main tutorial (§3.1.2).
- This replaces the need for a notebook-specific testing mechanism (nbval, pytest-notebook) — the "holes" pattern makes those inadequate anyway.

**Updated Phase 2 task sequence (early tasks):**
1. Fix `test_reproduction` → all tests green (§2.3 item 5)
2. Write exhaustive feature tests (expanded §2.3 item 11)
3. Test directory restructuring (§2.3 item 10)
4. Then major refactors...

**`sandbox.ipynb` — handled by developer.** Clear outputs and add to `.gitignore` (not currently gitignored).

**Corresponding tasks in PLAN.md:** P2.02 (exhaustive feature tests), P2.03 (notebook deletions, sandbox cleanup)

#### §3.3 Feature fixes (Phase 4 scope)

_Sources: PLAN.md §Phase 4, global CLAUDE.md §Cross-Package Issues #9, environment/CLAUDE.md §Known Issues #2 #6 and §Structural Questions #2, simulator/CLAUDE.md §Known Issues #8, tests/CLAUDE.md §Known Issues #5, interface/CLAUDE.md §Structural Questions #1, conf/CLAUDE.md §Known Issues #5, controllers/CLAUDE.md §Test Coverage (Logger)_

##### §3.3.1 Reproduction component — AGREED

**Decision: Fix the index bug (Phase 2, already agreed in §2.3 item 5). Add targeted tests. Mark as "functional but lightly tested" in docs. No heavy investment.**

- **Root cause of test failure:** Index mismatch in `reproduction/component.py` — `entity_type_energy` is built by selecting only entities of the target type, but the code indexes into it using global entity indices (`entity_type_idx`) instead of component-specific indices. Same issue for `recover_time`. Fix at ~4 locations.
- **Add 4 targeted tests:** (1) birth triggers when energy exceeds threshold, (2) death triggers when energy drops below threshold, (3) recovery time prevents immediate re-reproduction, (4) non-existing entities don't reproduce.
- Once the bug is fixed, the component logic (birth/death by energy thresholds) is straightforward. The exhaustive feature tests (§2.3 item 11) will also exercise reproduction indirectly if any session uses it.
- Heavy investment (fuzzing edge cases, multi-type reproduction) has poor ROI for this release.

**Corresponding tasks in PLAN.md:** P2.01 (fix + tests), P3.01 (document as "functional but lightly tested")

##### §3.3.2 Consumption component — AGREED

**Decision: No Phase 4 action. Keep as-is, document as "functional, tested indirectly."**

- No known bugs or test failures.
- Exercised indirectly by integration tests (`test_environments.py` runs fishing and non_transitive scenes which include consumption).
- The exhaustive feature tests (§2.3 item 11) will add coverage if sessions use consumption settings.
- Isolated unit testing (crafting spatial configurations, verifying matrix values) is high effort, low return given no reported issues.
- "Feeding matrix computation may have edge cases" was a generic audit concern, not a concrete bug.
- Defer isolated testing post-release.

**Corresponding tasks in PLAN.md:** No action needed

##### §3.3.3 Server-side recording — AGREED

**Decision: Remove broken code in Phase 2 (§2.1, already agreed). Design and implement a proper server-side recording mechanism in Phase 4.**

- The broken recording code (`record`, `start_recording`, `stop_recording`, `save_records`, `load` in `simulator.py`) is removed as dead code in §2.1.
- **Replacement:** A proper server-side recording component that records data into a dedicated JAX state field. This serves both the "recording" need (state capture over time for analysis/replay) and the "headless logging" need (§3.3.6) — researchers running headless simulations get their data from the JAX state.
- The component-based approach is a promising direction but needs a dedicated design discussion before implementation, at the start of Phase 4 work on this item — by then the codebase will be stabilized (component move, rigid body removal, etc.).
- Client-side `Logger` (in controllers) stays as-is for the notebook/programmatic use case.

**Corresponding tasks in PLAN.md:** P2.06 (remove broken recording code), P4.00 (server-side recording component)

##### §3.3.4 `render.py` — AGREED

**Decision: Fix bugs in Phase 2 (§2.2, already agreed). Defer "render from recorded data" capability to Phase 4 as part of §3.3.3 recording component scope.**

- Two rendering implementations exist: Bokeh (real-time in Panel UI) and matplotlib (`render.py`). Different libraries but similar logic (plotting entities in 2D).
- "Launching Bokeh headlessly" was imprecise — the actual need is rendering recorded data a posteriori (e.g. producing videos from headless simulation recordings).
- This depends on what the recording component (§3.3.3) stores, so rendering-from-recording should be designed alongside recording, not independently.
- `render.py` stays as a quick matplotlib tool for visual checks in notebooks/scripts. The proper video pipeline comes with the recording feature in Phase 4.

**Corresponding tasks in PLAN.md:** P2.09 (render.py bug fixes), P4.01 (render from recorded data)

##### §3.3.5 Braitenberg sensing/motor extensions — AGREED

**Decision: Phase 4, time permitting. Two concrete extensions identified. Not blocking for release.**

- **N evenly-spaced sensor cones:** Generalize proximeters from 2 fixed sensors (left/right) to an arbitrary number of regularly spaced sensor cones. Touches `sensorimotor.py` (sensor geometry, motor mapping) and likely the behavior system (behavior functions currently assume 2 inputs). Bounded but non-trivial.
- **Non-occluding proximeters:** Currently, a proximeter senses the closest entity in its cone regardless of subtype — non-target entities can shadow target entities. Add a non-occluding mode where the proximeter senses the closest entity of the target subtype, ignoring non-target entities in the way.
- The current 2-proximeter occluding setup works and is well-exercised by sessions 1-4 and tests. No audience journey is blocked.
- The original PLAN.md bullet "extend Braitenberg sensing/motor abilities" is narrowed to these two items.

**Corresponding tasks in PLAN.md:** P4.03

##### §3.3.6 Headless logging — merged into §3.3.3

**Corresponding tasks in PLAN.md:** P4.00 (via §3.3.3)

##### §3.3.7 `MaskFunction` extensibility — AGREED

**Decision: Defer to post-release. No action for this release.**

- Only one consumer (`BraitenbergComponent`), only one label (`'exists'`). Current behavior is correct for all scenes.
- Making it extensible is a design question tied to future use cases that don't exist yet.
- If §3.3.5 extensions (N sensors, non-occluding proximeters) need richer masking, that's the right time to generalize — design for the concrete need, not speculatively.

**Corresponding tasks in PLAN.md:** D.11

##### §3.3.8 Config validation — AGREED

**Decision: Defer to post-release. No action for this release.**

- Developer-facing guardrail, not a user-facing bug. Scene configs are written by developers, not end users.
- All current scenes have correct `subtype_labels` — no active failure.
- The developer tutorial (§3.1.4) will document the `subtype_labels` requirement, which is the more impactful fix for this release.

**Corresponding tasks in PLAN.md:** D.12

##### §3.3.9 Boolean `exists` field — AGREED

**Decision: Phase 2 task. Change `state.entity_state.exists` from int (0/1) to boolean.**

- Current int representation is confusing and adds unnecessary complexity (e.g. `exists == 1` instead of just `exists`).
- Preliminary scan shows ~15-20 locations across environment code, tests, and configs, but **this is not exhaustive** — a careful full-codebase analysis is required before making the changes.
- Common patterns include (maybe not exhaustive): `exists == 1` → `exists`, `exists == 0` → `~exists`, `.at[idx].set(1)` → `.at[idx].set(True)`, `dtype=int` → `dtype=bool`. No `exists * value` arithmetic patterns found in preliminary scan, but this must be verified.
- **Timing:** To decide, but potentially after rigid body removal (§2.1a), before exhaustive feature tests (§2.3 item 11), so tests are written against the clean API.

**Corresponding tasks in PLAN.md:** P2.08

---

#### §3.4 Minor cleanup items

_Sources: environment/CLAUDE.md §Known Issues #7 #9 and §Structural Questions #3, controllers/CLAUDE.md §Known Issues #4 #6 and §Known Issues Medium #3 #4 #5 and §Refactoring, interface/CLAUDE.md §Known Issues #7 #8 #9 #10 #11, simulator/CLAUDE.md §Known Issues #8 #9, utils/CLAUDE.md §Known Issues #3 #6 #7 #8 and §Refactoring, scripts/CLAUDE.md §Known Issues #4 #6 #7, conf/CLAUDE.md §Known Issues #4, notebooks/CLAUDE.md §Known Issues #9, global CLAUDE.md §Cross-Package Issues #4_

Lower-priority items that could be batched into a single cleanup pass:

**Code quality:**

##### §3.4.1 Missing `__all__` — AGREED

**Decision: Defer to post-release. No impact on functionality or user journeys.**

**Corresponding tasks in PLAN.md:** D.01

##### §3.4.2 Missing docstrings — AGREED

**Decision: Add docstrings to main public classes/methods in Phase 3 (alongside documentation). Internal helpers deferred — may be done before release, to decide later.**

- Main public classes/methods: `Controller.to_deal_with()`, `Controller.remote_to_ctrl()`, etc. (controllers/CLAUDE.md §Known Issues #6); `WindowManager.__init__()`, `create_interfaces()`, `ParameterizedData` methods (interface/CLAUDE.md §Known Issues #10)
- Internal helpers (deferred): `_get_ssl_context`, `_version_is_newer`, `_build_defaults_manifest` (utils/CLAUDE.md §Known Issues #8)

**Corresponding tasks in PLAN.md:** P3.10

##### §3.4.3 Missing type hints — AGREED

**Decision: Same as §3.4.2 — add type hints to main public classes in Phase 3, internal code deferred. May do before release, to decide later.**

- Main public classes: `Controller`, `RoutineHandler`, `BehaviorHandler` (controllers/CLAUDE.md §Refactoring)
- Internal code (deferred): `handle_server_interface.py` (utils/CLAUDE.md §Refactoring)

**Corresponding tasks in PLAN.md:** P3.10

##### §3.4.4 `parameterized.py` rename — RESOLVED

**Already done. TODO removed by developer.**

**Corresponding tasks in PLAN.md:** No action (already resolved)

##### §3.4.5 Repetitive controller/interface code — AGREED

**Decision: Defer to post-release. Works, just verbose. Abstraction design has regression risk.**

- Each entity controller maps user-facing property names (e.g. `left_motor`, `x_position`, `subtype`) to state paths via `__getattr__`/`__setattr__`. The routing logic branches on whether the field is in `entity_state`, a component-specific field, a controller parameter, or a split attribute (e.g. `left_motor` → `motor[0]`).
- This logic is spread across `EntityWrapper`, `EntityController`, `AgentController`, `WallController`, each adding its own layer. `AttributeMapping` exists as a helper but doesn't eliminate the routing complexity.
- A declarative field registry could reduce duplication, but designing it properly (handling all branch cases) is non-trivial and risks regressions. No user is affected by the current verbosity.

**Corresponding tasks in PLAN.md:** D.02

##### §3.4.6 "Component" naming ambiguity — AGREED

**Decision: No rename. Clarify in developer tutorial (§3.1.4).**

- A component is defined by its `Component` subclass (JAX step function). It optionally has a `Controller` (client-side API) and `Interface` (Panel UI). Simpler components (reset, friction, energy) have only the `Component` class.
- This is consistent with the current naming and code structure. The developer tutorial should explain this clearly.

**Corresponding tasks in PLAN.md:** P3.06 (documented in developer tutorial)

**Design smells:**

##### §3.4.7 `apply_changes()` ownership — AGREED

**Decision: No refactoring needed. Remove the misleading TODO. Current separation is correct.**

- The TODO suggests duplication, but the three layers have distinct responsibilities:
  1. `VivariumController.apply_changes()` — collects pending changes from the `Remote` proxy, delegates to client, handles `close` signal.
  2. `SimulatorGRPCClient.set_changes()` — network layer: serializes to protobuf, sends over gRPC, optionally fetches back updated state.
  3. `Simulator.set_changes()` — server-side: applies changes to dataclass, syncs simulator attributes, manages run/stop state.
- The only minor smell is the streaming check (`hasattr(self.client, 'is_streaming')`) in `apply_changes()`, which is a leaky abstraction — but not worth refactoring for this release.

**Corresponding tasks in PLAN.md:** P2.03 (remove misleading TODO)

##### §3.4.8 `RoutineHandler`/`BehaviorHandler` similarity — AGREED

**Decision: Defer base class extraction to post-release. Rename `utils.py` → `handlers.py` in Phase 2 (after `kill_session` removal in §2.1).**

- ~80% of code is duplicated (registry pattern: attach/detach/start/stop/print, two-dict pattern, threading lock, name resolution). Only `behave()` (weighted motor blending) and `routine_step()` (direct call + error auto-removal) differ.
- A base class would be clean but is low priority — internal code, works correctly, no divergence risk.
- The module name `utils.py` is misleading — it contains specific, well-defined classes (`Logger`, `RoutineHandler`, `BehaviorHandler`), not generic utilities. Rename to `handlers.py` after `kill_session` removal leaves only these three classes.

**Corresponding tasks in PLAN.md:** P2.06 (rename utils.py → handlers.py), D.03 (base class extraction deferred)

##### §3.4.9 Implicit step routing — AGREED

**Decision: Document in Phase 3. No code change needed.** The mechanism works correctly — the main tutorial and developer guide should clarify the multi-client stepping model (controllers/CLAUDE.md §Known Issues Medium #4).

**Corresponding tasks in PLAN.md:** P3.04 (documented in main CS tutorial)

##### §3.4.10 `BehaviorHandler.behave()` tight coupling — AGREED

**Decision: Document in Phase 3. No code change.** The coupling is intentional — behaviors set motor values on agents. `BehaviorHandler` is only ever used with `AgentController` (controllers/CLAUDE.md §Known Issues Medium #5).

**Corresponding tasks in PLAN.md:** P3.04 (documented in main CS tutorial)

##### §3.4.11 `scene_name` setter convention — AGREED

**Decision: Quick fix in Phase 2 — remove the setter, let Python's default behavior raise `AttributeError`.** Trivial one-liner (simulator/CLAUDE.md §Known Issues #8).

**Corresponding tasks in PLAN.md:** P2.03

##### §3.4.12 `load()` as standalone function — RESOLVED

**Already covered: `load()` is part of the broken recording code being removed in §2.1.** (simulator/CLAUDE.md §Known Issues #9)

**Corresponding tasks in PLAN.md:** No action (covered by P2.06)

##### §3.4.13 Platform PID functions visibility — AGREED

**Decision: Defer to post-release. Purely cosmetic — no external consumers.** The module name (`handle_server_interface.py`) is also vague, but renaming it now doesn't help — it will be addressed naturally if/when the module is split (utils/CLAUDE.md §Known Issues #6).

**Corresponding tasks in PLAN.md:** D.04

##### §3.4.14 Thread safety in Panel update loop — AGREED

**Decision: No locking changes for this release. Add explanatory comments in Phase 2 documenting the threading risk.**

- `update_plot_cb()` reads `controller.client.state` with no locking while gRPC callbacks update it from another thread. CPython's GIL makes individual attribute assignments atomic, so torn reads are unlikely. But `set_changes()` updates `state` and `controller_parameters` in two separate assignments — the update callback could read between them, causing brief visual inconsistencies (new positions, old colors).
- This may have caused rendering inconsistencies observed in the past. Adding comments near `update_plot_cb()` and the gRPC client state assignment will help future debugging.
- Proper fix (locking or atomic state+params swap) risks introducing deadlocks — defer to post-release (interface/CLAUDE.md §Known Issues #7).

**Corresponding tasks in PLAN.md:** P2.03 (explanatory comments), D.13 (proper fix deferred)

##### §3.4.15 Hardcoded scene enumeration — AGREED

**Decision: Defer to post-release.** Current scene set is stable for the release.

- `get_available_scenes()` categorizes scenes by name pattern (`startswith('session')`, etc.) with `research` as the catch-all default. Adding a new category or moving a scene requires editing the function. Brittle but functional (utils/CLAUDE.md §Known Issues #3).

**Corresponding tasks in PLAN.md:** D.05

##### §3.4.16 Implicit client inclusion — AGREED

**Decision: Defer to post-release. Not a bug — discoverability concern addressed by developer tutorial (§3.1.4) explaining the config inheritance chain** (conf/CLAUDE.md §Known Issues #4).

**Corresponding tasks in PLAN.md:** D.06

##### §3.4.17 Global state in `run_interface.py` — AGREED

**Decision: Defer to post-release.** Script entry point, not a library module — global state is acceptable here (scripts/CLAUDE.md §Known Issues #7).

**Corresponding tasks in PLAN.md:** D.07

##### §3.4.18 Notebook execution order — AGREED

**Decision: Defer to post-release.** Pedagogical concern, not technical. Sessions are numbered, explicitly reference previous sessions when a concept was introduced earlier, and the tutorial (§3.1.2) will explain the progression. Code-level enforcement would be fragile and over-engineered (notebooks/CLAUDE.md §Known Issues #9).

**Corresponding tasks in PLAN.md:** D.08

**Hardcoded values:**

##### §3.4.19 Hardcoded Jupyter port — AGREED

**Decision: Quick fix in Phase 2. Extract a named constant, replace ~7 occurrences.**

- `8889` appears as default parameter in 6 function signatures and one fallback assignment in `panel_app.py:729` (which is already overridden by config if `interface.notebook.jupyter_port` is set).
- All functions already accept port as a parameter — no config threading needed. Just extract `DEFAULT_JUPYTER_PORT = 8889` in one place (e.g. `runtime.py`) and import it elsewhere (interface/CLAUDE.md §Known Issues #8, utils/CLAUDE.md §Known Issues #7).

**Corresponding tasks in PLAN.md:** P2.13

**File locations:**

##### §3.4.20 `rthook_jupyter_matplotlib.py` location — AGREED

**Decision: Move into `vivarium/runtime/` during §1.4a refactoring. No separate action needed.**

- It's a PyInstaller runtime hook (executed via `exec()` at startup), not a user-facing entry point — `scripts/` is the wrong semantic home. `vivarium/runtime/` (the PyInstaller/deployment package created in §1.4a) is the natural location (scripts/CLAUDE.md §Known Issues #4).

**Corresponding tasks in PLAN.md:** P2.13

##### §3.4.21 Ngrok extraction — AGREED

**Decision: No action needed. Already optional.**

- `pyngrok` is in `extras_require["colab"]` in `setup.py`, not in `install_requires`. Code imports lazily with a clear error message if not installed. When §1.4a splits `handle_server_interface.py`, ngrok functions will naturally become their own module in `vivarium/runtime/` (utils/CLAUDE.md §Refactoring).

**Corresponding tasks in PLAN.md:** No action (already optional, handled naturally by P2.13)

**Nice-to-have features:**

##### §3.4.22 `--version` flag — AGREED

**Decision: Defer to post-release.** `get_version()` exists in `runtime.py` — adding the flag is trivial but not needed for any audience journey (scripts/CLAUDE.md §Known Issues #6).

**Corresponding tasks in PLAN.md:** D.09

##### §3.4.23 Checksum validation — AGREED

**Decision: Defer to post-release.** PyInstaller-specific, requires server-side changes too (utils/CLAUDE.md §Known Issues #5).

**Corresponding tasks in PLAN.md:** D.10

**Already resolved:**

##### §3.4.24 RigidBody — already resolved

- ~~RigidBody support is live but dormant~~ — **Reversed: removing entirely (see §2.1a)**

**Corresponding tasks in PLAN.md:** No action (reversed — P2.07 removes entirely)

---

## Currently Discussing

_(All topics discussed.)_

---

## Remaining to Discuss

_(None — all topics have been discussed and agreed.)_
