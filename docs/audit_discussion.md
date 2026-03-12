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

#### §1.6 Script entry points: redundancy and cleanup

_Sources: scripts/CLAUDE.md §Known Issues #3 #5, utils/CLAUDE.md §Known Issues #2_

**Decision: Remove `run_vivarium.py` entirely.**

- `run_interface.py` already handles the full workflow: scene selection UI → server startup → interface. It has proper cleanup (signal handlers, atexit, process killing).
- `run_vivarium.py` is **not** in the PyInstaller `.spec` file — the spec builds `vivarium-server`, `vivarium-interface`, and `vivarium-jupyter` from their respective scripts.
- Not used programmatically — `handle_server_interface.py` spawns `run_server.py` and `run_interface.py` directly.
- No test coverage. The 4 layers of PyInstaller guards (spawn counter, env vars, child process detection) are dead code since it's never bundled.
- Removes the inconsistent scene argument style issue (positional arg vs Hydra override).

---

## Currently Discussing

_(see next item below)_

---

## Remaining to Discuss

Topics are ordered by layer (architectural first, then cleanup, then docs/features) and within each layer by importance. Issues that are symptoms of the same root cause are grouped together.

### Layer 2 — Per-Package Cleanup

#### 2.1 Dead code removal

_Sources: global CLAUDE.md §Dead Code to Remove and §Cross-Package Issues #9, environment/CLAUDE.md §Known Issues (is_entity_component, rigid_body_to_point_particle), simulator/CLAUDE.md §Known Issues #3 #6 #7 #8, interface/CLAUDE.md §Known Issues #3, controllers/CLAUDE.md §Known Issues #5 #6, scripts/CLAUDE.md §Known Issues #1 #2, conf/CLAUDE.md (session_6.yaml), notebooks/CLAUDE.md §Known Issues #1_

The audit identified dead code across every package. Listed here grouped by kind:

**Dead files:**
- `vivarium/environment/components/eco_evo/component.py` — empty file (0 bytes)
- `scripts/print_config.py` — unused utility
- `scripts/profiling.py` — uses non-existent `SceneConfiguration` import
- `conf/scene/session_6.yaml` — outdated, no `scene_name`, untested
- `notebooks/sessions/session_5_logging copy.ipynb` — duplicate backup (290KB)

**Dead code in simulator:**
- `nested_fields_to_access` dict + commented-out decorator reference (simulator.py:36-43, simulator_client.py:24)
- `SetState` RPC handler calling non-existent `simulator.set_state()` (simulator_server.py:112-118)
- Recording feature (`record`, `start_recording`, `stop_recording`, `save_records`, `load`) — marked "probably broken" (simulator.py)

**Dead code in interface:**
- `streaming_toggle` widget (panel_app.py:687) — created but never displayed
- `streaming_toggle_cb()` (panel_app.py:972) — defined but never registered
- `self.notebook_mode` (panel_app.py:84) — set but never read
- Commented-out config_update logic (panel_app.py:993-994)

**Dead code in environment:**
- `is_entity_component` flag in `component.py:13` and `entities/component.py:29` — set but never read
- `rigid_body_to_point_particle()` in `utils.py:50-91` — marked deprecated, only referenced in sandbox notebook

**Dead code in controllers:**
- `kill_session()` — only referenced in outdated `session_6_bonus.ipynb`
- Questionable `__init__.py` exports (`set_nested_attr`, `Controller`, `AttributeMapping` — no external consumer via `__init__`)

**To discuss:** Any of these controversial to remove? The recording feature and bidirectional streaming are the biggest decisions (remove vs. fix). Everything else is straightforward deletion.

#### 2.2 Bugs to fix

_Sources: environment/CLAUDE.md §Known Issues #2 #4, interface/CLAUDE.md §Known Issues #4 #6, global CLAUDE.md §Patterns to Fix #6 #7, simulator/CLAUDE.md §Known Issues #6 #7, utils/CLAUDE.md §Known Issues #2, conf/CLAUDE.md §Known Issues #2 #3 #5_

**Rendering bugs:**
- `render.py:18` — `diameter[idx][exists][exists]` double-indexes with `[exists]` (environment/CLAUDE.md §Known Issues #2)
- `render.py:41-42` — same double-indexing pattern for orientation
- `render.py:67` — `plt.xlim(0, box_size)` called twice (should be `xlim` then `ylim`)

**Typos and code smells:**
- `udpate_other_interfaces` typo in both `environment/components/interface.py:65` and `interface/panel_app.py:645` (environment/CLAUDE.md §Known Issues #4, interface/CLAUDE.md §Known Issues #4)
- Bare `except:` in `utils/handle_server_interface.py:469` — catches KeyboardInterrupt (global CLAUDE.md §Patterns to Fix #7)

**Interface issues:**
- `_start_server_cb()` catches exceptions but only updates status UI — no `lg.exception()` for stack traces (interface/CLAUDE.md §Known Issues #6)

**Config issues:**
- `braitenberg.yaml` and `particle_lenia.yaml` missing `_self_` in defaults list — causes Hydra UserWarning (conf/CLAUDE.md §Known Issues #2)
- `demo.yaml` has 60+ lines of commented-out code (conf/CLAUDE.md §Known Issues #3)

**Simulator issues:**
- `to_config()` uses fragile hardcoded relative path `'../../conf/scene/simulator'` (simulator/CLAUDE.md §Known Issues #6)
- Missing error handling in gRPC RPC handlers — opaque 500 errors (simulator/CLAUDE.md §Known Issues #7)

**Utils issues:**
- `start_server_and_interface()` imported by some notebooks but doesn't exist (utils/CLAUDE.md §Known Issues #2)

**To discuss:** All straightforward fixes — any we should skip for this release?

#### 2.3 Test suite cleanup and gaps

_Sources: tests/CLAUDE.md §Known Issues #1-10 and §Coverage Analysis and §Refactoring Opportunities, environment/CLAUDE.md §Test Coverage, controllers/CLAUDE.md §Test Coverage, simulator/CLAUDE.md §Test Coverage, interface/CLAUDE.md §Test Coverage, utils/CLAUDE.md §Test Coverage, conf/CLAUDE.md §Test Coverage_

**Test quality issues to fix:**
- No-op tests: `test_instantiate()` is just `pass`, `test_braitenberg()` has no assertions (tests/CLAUDE.md §Known Issues #1)
- Duplicate `test_remote()` function names in `test_dataclass_api.py` (tests/CLAUDE.md §Known Issues #2)
- Commented-out test `test_is_connected_verify_detects_no_server` (tests/CLAUDE.md §Known Issues #4)
- Incomplete tests with TODO comments (tests/CLAUDE.md §Known Issues #3)
- `test_reproduction` pre-existing failure (tests/CLAUDE.md §Known Issues #5)

**Coverage gaps (untested core logic):**
- Individual physics components: collision, friction, reset (tests/CLAUDE.md, environment/CLAUDE.md §Test Coverage)
- Braitenberg sensorimotor in isolation (environment/CLAUDE.md §Test Coverage, tests/CLAUDE.md §Not Tested)
- Consumption matrix, energy distribution, wall geometry (environment/CLAUDE.md §Test Coverage)
- ProximityMapComponent — exercised via conftest fixture but no dedicated tests (environment/CLAUDE.md §Test Coverage)
- Particle Lenia — only via scene tests, no direct tests (tests/CLAUDE.md §Not Tested)
- `render.py` — has known bugs and no tests (environment/CLAUDE.md)
- Logger, RoutineHandler, BehaviorHandler (controllers/CLAUDE.md §Test Coverage)
- Controller base class, error handling paths (controllers/CLAUDE.md §Test Coverage)
- SimulatorController (simulator/CLAUDE.md via tests/CLAUDE.md §Not Tested)
- Streaming rate limiting (simulator/CLAUDE.md §Test Coverage)
- Panel app interactive features and all component interfaces (interface/CLAUDE.md §Test Coverage, tests/CLAUDE.md §Not Tested)
- `scene_configs.py` partial: `get_available_scenes`, position/orientation generators, `compute_parameters` untested (utils/CLAUDE.md §Test Coverage)
- `handle_server_interface.py`: Jupyter lifecycle, ngrok, port killing, process PID lookup untested (utils/CLAUDE.md §Test Coverage)
- `runtime.py`: command builders (`get_server_command`, etc.) untested (utils/CLAUDE.md §Test Coverage)
- Small utils modules: `converters.py`, `timer.py`, `jax_utils.py` — no dedicated tests, exercised indirectly (utils/CLAUDE.md §Test Coverage, tests/CLAUDE.md §Not Tested)
- 3 research scene configs (`demo`, `excretion`, `boyds`) have no test coverage — low effort to add to `test_environments.py` parametrize (conf/CLAUDE.md §Test Coverage)

**Structural improvements:**
- No test markers (`@pytest.mark.slow`, `@pytest.mark.unit`) (tests/CLAUDE.md §Known Issues #9)
- No fixture documentation in conftest.py (tests/CLAUDE.md §Known Issues #10)
- Hardcoded `NUM_STEPS` varies across 4 test files (4, 5, 6, 10) — inconsistent (tests/CLAUDE.md §Known Issues #6)
- NaN workaround in `test_environments.py` (lines 27-29 revert state on NaN) — indicates potential numerical instability in physics (tests/CLAUDE.md §Known Issues #7)
- Test files don't mirror source package structure (tests/CLAUDE.md §Known Issues #8)

**To discuss:** What's the testing strategy for the release? Fix the broken/no-op tests (high value, low effort). Adding markers and fixture docs (medium effort). Adding missing component tests (high effort) — worth it for the release or defer?

---

### Layer 3 — Documentation & Feature Scope

#### 3.1 Audience readiness and documentation plan

_Sources: global CLAUDE.md §Audience Journey Readiness #1-3, PLAN.md §Phase 3, notebooks/CLAUDE.md §Known Issues #2-3 and §Structural Questions #1 #3 #4, environment/CLAUDE.md §Structural Questions #2 (render.py as headless output), interface/CLAUDE.md §Structural Questions #1 (Bokeh headless). Also resolves §1.2 (config–code structure)._

**Researchers (headless JAX):** Blocked by documentation. No current docs or working notebooks. Server-side notebooks all use obsolete import paths. Need at minimum: one working notebook showing `Environment` / `Simulator` headless usage with current API.

**CS students (programmatic control):** Mostly ready. Sessions 1-4 are excellent. Gaps:
- Session 5 (logging) incomplete
- Session 6 (eco-evo) uses deprecated API, needs rewrite
- `quickstart_tutorial.ipynb` outdated (uses `NotebookController`)
- No standalone API reference (sessions are progressive, not reference)

**Web interface (younger students):** Functional but poorly documented. Only `web_interface_tutorial.md` exists. `WindowManager` is a brittle monolith with minimal tests.

**Developers (extending the code):** No "how to add a new component" guide. Critical for incoming Master's student (mentioned in PLAN.md Phase 3).

**To discuss:** For each audience, what's the minimum viable documentation for this release? Specifically:
- Researchers: write one new notebook, or rewrite an existing server-side one? Better to not rely on config files? (i.e. using only main constructors)
- CS students: fix sessions 5-6, rewrite quickstart, or write a standalone API reference?
- Web interface: is the existing markdown tutorial sufficient?
- Developers: is the component guide in scope for this release?

**Also:** Do lightweight audience journey sketches here to resolve §1.2 (config–code structure — document-only or refactor?).

#### 3.2 Notebook decisions

_Sources: notebooks/CLAUDE.md §Known Issues #4 #6 #7 and §Structural Questions #2 #3 #4, tests/CLAUDE.md §Structural Questions #2_

Several notebook-related decisions are deferred from the audit:

- **Tutorials folder:** All three notebooks are outdated. Update `quickstart_tutorial.ipynb` as an API entry point? Write a new one? Or archive the folder? (notebooks/CLAUDE.md §Known Issues #7)
- **Session 6 scope:** `miniproject_template.ipynb` already covers most of session 6's content (eco-evo dynamics, custom configs). Rewrite session 6 as a lighter complement, or merge into miniproject? (notebooks/CLAUDE.md §Structural Questions #4)
- **Notebook testing:** No mechanism to verify notebooks work after code changes. Options: reference-solution notebooks, metadata-marked skip cells, or test only non-session notebooks. (tests/CLAUDE.md §Structural Questions #2)
- **`sandbox.ipynb`:** 1.9M with embedded outputs. Clear outputs or gitignore? (notebooks/CLAUDE.md §Known Issues #6)
- **`google_colab.ipynb`:** Hardcoded branch reference, incomplete TODOs. Fix or remove? (notebooks/CLAUDE.md §Known Issues #4)
- **Server-side notebooks:** Archive, remove, or rewrite one for the researcher audience? (notebooks/CLAUDE.md §Structural Questions #3)

#### 3.3 Feature fixes (Phase 4 scope)

_Sources: PLAN.md §Phase 4, global CLAUDE.md §Cross-Package Issues #9, environment/CLAUDE.md §Known Issues #2 #6 and §Structural Questions #2, simulator/CLAUDE.md §Known Issues #8, tests/CLAUDE.md §Known Issues #5, interface/CLAUDE.md §Structural Questions #1, conf/CLAUDE.md §Known Issues #5, controllers/CLAUDE.md §Test Coverage (Logger)_

From PLAN.md Phase 4 rough scope, informed by audit findings:

- **Reproduction component:** Has a pre-existing test failure. Complex state-dependent birth/death logic with no dedicated tests. Fix and test, or mark as experimental?
- **Consumption component:** Untested in isolation. Feeding matrix computation may have edge cases.
- **Recording feature:** Currently marked broken in simulator. Remove entirely (simplest), or fix and test?
- **`render.py`:** Has bugs (double-indexing, xlim/ylim). Fix the bugs, or replace with something better? The Bokeh renderers in the Panel UI already know how to visualize entities — could potentially be used headlessly to replace matplotlib-based `render.py`, but launching Bokeh headlessly is non-trivial (interface/CLAUDE.md §Structural Questions #1, environment/CLAUDE.md §Structural Questions #2).
- **Braitenberg sensing/motor extensions:** Mentioned in PLAN.md but not elaborated. What specifically?
- **Headless logging:** No logging infrastructure for headless simulation (researchers need this). `Logger` in controllers is client-side only.
- **`MaskFunction` extensibility:** Currently only supports `'exists'` label, hardcoded in `environment.py`. Works for current use cases but limits future flexibility (environment/CLAUDE.md §Known Issues #6).
- **Config validation:** No validation that scene-defined entity subtypes match `subtype_labels` — UI controllers may fail silently (conf/CLAUDE.md §Known Issues #5).

**To discuss:** Which of these are realistic for this release? Which should be explicitly deferred?

#### 3.4 Minor cleanup items

_Sources: environment/CLAUDE.md §Known Issues #7 #9 and §Structural Questions #3, controllers/CLAUDE.md §Known Issues #4 #6 and §Known Issues Medium #3 #4 #5 and §Refactoring, interface/CLAUDE.md §Known Issues #7 #8 #9 #10 #11, simulator/CLAUDE.md §Known Issues #8 #9, utils/CLAUDE.md §Known Issues #3 #6 #7 #8 and §Refactoring, scripts/CLAUDE.md §Known Issues #4 #6 #7, conf/CLAUDE.md §Known Issues #4, notebooks/CLAUDE.md §Known Issues #9, global CLAUDE.md §Cross-Package Issues #4_

Lower-priority items that could be batched into a single cleanup pass:

**Code quality:**
- Missing `__all__` in several `__init__.py` files (environment/CLAUDE.md §Known Issues #9, interface/CLAUDE.md §Known Issues #9)
- Missing docstrings on base class methods: `Controller.to_deal_with()`, `Controller.remote_to_ctrl()`, etc. (controllers/CLAUDE.md §Known Issues #6); `WindowManager.__init__()`, `create_interfaces()`, `ParameterizedData` methods (interface/CLAUDE.md §Known Issues #10); helper functions `_get_ssl_context`, `_version_is_newer`, `_build_defaults_manifest` (utils/CLAUDE.md §Known Issues #8)
- Missing type hints on `Controller`, `RoutineHandler`, `BehaviorHandler` (controllers/CLAUDE.md §Refactoring), `handle_server_interface.py` (utils/CLAUDE.md §Refactoring)
- `parameterized.py` has a TODO to rename the file (interface/CLAUDE.md §Known Issues #11)
- Controller/interface code is repetitive across entity types — could benefit from declarative field mapping (environment/CLAUDE.md §Known Issues #7)
- "Component" naming ambiguity — means both the full triad (JAX + controller + interface) and just `component.py`. A documentation concern, deferred to Phase 3 (environment/CLAUDE.md §Structural Questions #3)

**Design smells:**
- `apply_changes()` has a TODO questioning ownership — duplicates logic from `SimulatorGRPCClient.set_changes()` (controllers/CLAUDE.md §Known Issues #4)
- `RoutineHandler` and `BehaviorHandler` are similar but separate — could share a base class (controllers/CLAUDE.md §Known Issues Medium #3)
- Step routing logic is implicit: whether this client drives stepping depends on `simulator.run_from == client.name` — not documented (controllers/CLAUDE.md §Known Issues Medium #4)
- `BehaviorHandler.behave()` assumes agent is an `EntityController` with `__setattr__` recording — tight coupling, undocumented contract (controllers/CLAUDE.md §Known Issues Medium #5)
- `scene_name` property setter raises `AttributeError` unconventionally (simulator/CLAUDE.md §Known Issues #8)
- `load()` should be a standalone function, not a Simulator method (simulator/CLAUDE.md §Known Issues #9)
- Platform-specific PID functions should be internal (`_`-prefixed) (utils/CLAUDE.md §Known Issues #6)
- Thread safety concerns in Panel update loop (interface/CLAUDE.md §Known Issues #7)
- Scene enumeration is hardcoded in `get_available_scenes()` — uses string matching, new scene types require code changes (utils/CLAUDE.md §Known Issues #3)
- Implicit client inclusion: `clients/collision.yaml` is included via `base_physics.yaml` but never directly referenced by scene files (conf/CLAUDE.md §Known Issues #4)
- Global state in `run_interface.py` (`_cleanup_done`, `_window_manager`) — works but inelegant (scripts/CLAUDE.md §Known Issues #7)
- No notebook execution order enforcement — students could run session 3 before session 1 (notebooks/CLAUDE.md §Known Issues #9)

**Hardcoded values:**
- Jupyter port 8889 hardcoded in multiple places (interface/CLAUDE.md §Known Issues #8, utils/CLAUDE.md §Known Issues #7)

**File locations:**
- `rthook_jupyter_matplotlib.py` should live in a `pyinstaller_hooks/` directory per convention, not in `scripts/` (scripts/CLAUDE.md §Known Issues #4)
- Consider extracting ngrok to optional submodule (utils/CLAUDE.md §Refactoring)

**Nice-to-have features:**
- No `--version` flag on any script (scripts/CLAUDE.md §Known Issues #6)
- No checksum validation for downloaded updates in `updater.py` (utils/CLAUDE.md §Known Issues #5)

**Already resolved (no action needed for release):**
- RigidBody support is live but dormant — all physics components and entity controllers have conditional rigid body paths, but no scene uses it. Keep the code, don't invest in testing it (global CLAUDE.md §Cross-Package Issues #4, environment/CLAUDE.md §Structural Questions #1)
