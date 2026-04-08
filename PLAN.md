# Vivarium Release Plan

_Living document — updated at the end of every working session._
_Last updated: 2026-04-08_

---

## Goal

Release a clean, documented version of Vivarium demonstrating its three core use cases:

1. **Headless simulation in JAX** — for researchers comfortable with JAX
2. **Programmatic Pythonic control** — for CS students who know Python but not JAX
3. **Web interface** — for students with little or no programming background

_A fourth journey (component development, for developers extending the code) and the journey naming convention will be finalized in P3.00._


---

## General Guidelines

_These apply to every task below._

**Starting a task:**
1. Read the task description fully.
2. In parallel: (a) read all key files listed in the task and proactively analyze related code to understand the current state, and (b) ask the user if `pytest` should be run to verify the baseline (don't assume — if the previous task just confirmed green, it may be unnecessary).
3. Propose a branch name: `refactor-0.2.0/<task-id>-<short-name>` (e.g., `refactor-0.2.0/P2.03-early-cleanup`).
4. Inform the user of how you plan to proceed and iterate before executing. For complex tasks (structural changes, many files), use Plan mode.

**During execution:**
5. Only change what the task describes. If you spot an unrelated bug, flag it to the user and add it to the **Discovered Bugs** section below with a short description, location, and severity.
6. When writing or modifying docstrings, use **Google-style format** (not reST/Sphinx). This is the project standard.

**Reviewing:**
7. Present the changes to the user for review.
8. Iterate until the user agrees the task is complete.

**Completing a task:**
9. In parallel: (a) ask the user if `pytest` should be run, and (b) update affected CLAUDE.md files as specified in the task.
10. Remove the task in `PLAN.md` and add a summary at the end of the Completed section. 
11. Propose a commit message (unsigned).
12. The user reviews, commits, and handles the merge to `dev`.

**Phase gate (before starting Phase 3):**
- All Phase 2 tasks complete
- Full `pytest` — all green
- Full review of all CLAUDE.md files against actual codebase
- No dangling dependencies or TBD items in Phase 3

**Test directory structure:**
The `tests/` directory mirrors the source package structure (`tests/environment/`, `tests/simulator/`, `tests/controllers/`, `tests/interface/`, `tests/utils/`, `tests/scripts/`). When adding new tests or moving source files, keep the test directory mirroring the source packages.

**In general**
Before running the full test suite, first carefully consider if this is necessary or if it is sufficient to run a subset. The full suite takes quite a long time to execute. If you have a doubt, just ask.

---

## Phases

### Phase 1 — Codebase Audit ✓
_Completed. Per-package audits, cross-package synthesis, and planning discussion._

---

### Phase 2 — Cleanup & Refactoring

#### P2.11 — Dynamic dataclass fixes
- **Status:** [ ]
- **Dependencies:** P2.10 (component move — codebase structure stable)
- **Key files:** `vivarium/simulator/simulator.py`
- **CLAUDE.md updates:** `vivarium/simulator/CLAUDE.md` — update architecture section (ghost attributes eliminated, default controller_parameters in __init__).

Fix ghost attributes in `Simulator` by eliminating `update_from_dataclass()` and always accessing through `self.controller_parameters.simulator`. Build a default `controller_parameters` in `__init__` when none is provided.

**Steps:**
1. **Remove the `update_from_dataclass()` call** in `set_changes()` — this is what creates ghost attributes by copying `run_from`, `simulation_running`, and `freq` onto the Simulator instance.
2. **Replace direct `self.X` accesses** with `self.controller_parameters.simulator.X` for `run_from` and `simulation_running` (3 locations in `set_changes()` and `run()`).
3. **Fix `freq` aliasing** — `to_config()` uses `self.freq` but `__init__` stores `self._freq`. Change to access `self.controller_parameters.simulator.freq`.
4. **Build default `controller_parameters` in `__init__`** when none is provided, so the headless path (`Simulator(env=env, freq=10)`) and the config path work identically:
   ```python
   if controller_parameters is None:
       controller_parameters = create_dataclass_from_dict('ControllerParameters', {
           'simulator': {'freq': freq, 'scene_name': scene_name,
                        'run_from': 'server', 'simulation_running': False,
                        'client_names': []}
       })
   ```
5. **Remove `update_from_dataclass()` function** (top of simulator.py) — no longer needed.
6. **Drop misleading `self = ` assignment** in `set_changes()` — `update_dataclass_from_change_list` mutates the Simulator in place via `setattr`, so the return value is the same object. Replace with a plain call + comment:
   ```python
   # Mutates self.state and/or self.controller_parameters in place
   update_dataclass_from_change_list(self, changes)
   ```

Note: `nested_fields_to_access` dead code was already removed in P2.06.

#### P2.12 — Streaming cleanup
- **Status:** [ ]
- **Dependencies:** P2.11 (dynamic dataclass fixes — simulator stabilized)
- **Key files:** `vivarium/simulator/grpc_server/simulator_client.py`, `vivarium/simulator/grpc_server/simulator_server.py`, `vivarium/simulator/grpc_server/protos/simulator.proto`, `vivarium/interface/panel_app.py`, `vivarium/controllers/vivarium_controller.py`, `conf/scene/interface/base_interface.yaml`, `scripts/dev/benchmark_streaming_real.py`
- **CLAUDE.md updates:** `vivarium/simulator/CLAUDE.md` — update gRPC RPCs section (remove BidirectionalStep). `vivarium/interface/CLAUDE.md` — update architecture section (streaming always on, pending_state_update wired in).

Remove bidirectional streaming (dormant code), clean up dead UI code, and fix the pending state update flag.

**Remove (bidirectional streaming — fully implemented but no active code path uses it):**
- `bidirectional_step_sync()` and `bidirectional_step_generator()` in `simulator_client.py`
- `BidirectionalStep` RPC handler in `simulator_server.py` + proto definition (regenerate proto)
- `use_streaming` parameter in `VivariumController.start_controller_thread()`
- `scripts/dev/benchmark_streaming_real.py`

**Remove (dead UI code):**
- `streaming_toggle` widget + `streaming_toggle_cb` callback in `panel_app.py`
- `use_streaming` config key in `conf/scene/interface/base_interface.yaml` (streaming is always on in Panel UI)

**Fix:**
- Wire `_pending_state_update` flag into `update_plot_cb()` — skip repaint when no new state has arrived. The flag is already set by the streaming callback but never checked by the update loop.

**Keep as-is:**
- `StreamState` RPC, `start_state_stream()`/`stop_state_stream()`, `_start_streaming()`/`_stop_streaming()`
- `is_streaming` conditional in `apply_changes()` / `set_changes()` — correctly distinguishes Panel UI (streaming, fire-and-forget changes) from notebooks (no streaming, `SetChangesReturnsState`)

#### P2.13 — utils → runtime package split
- **Status:** [ ]
- **Dependencies:** P2.12 (streaming cleanup — simulator/interface stabilized)
- **Key files:** `vivarium/utils/handle_server_interface.py`, `vivarium/utils/runtime.py`, `vivarium/utils/updater.py`, `vivarium/utils/converters.py`, `vivarium/utils/__init__.py`, `scripts/rthook_jupyter_matplotlib.py`
- **CLAUDE.md updates:** `vivarium/utils/CLAUDE.md` — update structure, file table, and dependency graph (moved files removed, converters.py deleted). `CLAUDE.md` (global) — update package dependency map to include `vivarium/runtime/`. Create `vivarium/runtime/CLAUDE.md` with purpose, structure, and public API.

Create a new top-level `vivarium/runtime/` package. Split `handle_server_interface.py` into modules, move it there along with `runtime.py` (renamed `paths.py`) and `updater.py`. Clean up `converters.py`.

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

**Resulting `vivarium/utils/`:**
```
vivarium/utils/
  __init__.py
  scene_configs.py      # Hydra config loading, scene enumeration, template expansion
  dataclass_wrapper.py  # Remote proxy, change protocol, dataclass updates
  timer.py              # SleepTimer
  jax_utils.py          # JAX-MD dataclass detection
```

**Clean up `converters.py`:** Most functions are dead code (`upper_camel_to_snake`, `snake_to_upper_camel`, `class_import_path`, `import_class` — none used anywhere). Only `access_nested_fields` is live (used by `environment.py`). Move it to an appropriate location (e.g., inline in `environment.py` or keep in utils under a better name), then delete `converters.py`.

**Move `scripts/rthook_jupyter_matplotlib.py` → `vivarium/runtime/`** — it's a PyInstaller runtime hook, not a user-facing entry point. `vivarium/runtime/` (the deployment package) is its natural home.

**Extract `DEFAULT_JUPYTER_PORT = 8889`** in `vivarium/runtime/_jupyter.py` (or `paths.py`). Replace ~7 occurrences of the hardcoded `8889` across function signatures and `panel_app.py` with an import of this constant.

**Workflow:**
1. Create `vivarium/runtime/` package with `__init__.py`
2. Move and rename `utils/runtime.py` → `runtime/paths.py`
3. Move `utils/updater.py` → `runtime/updater.py`
4. Split `utils/handle_server_interface.py` into the 5 internal modules
5. Move `scripts/rthook_jupyter_matplotlib.py` → `vivarium/runtime/`
6. Extract `DEFAULT_JUPYTER_PORT` constant
7. Clean up `converters.py` (move `access_nested_fields`, delete file)
8. Update `runtime/__init__.py` to re-export public API
9. Update all imports across the codebase
10. Run tests

**Note from the developer:** It might be more efficient to leverage VSCode refactoring for moving modules/functions — VSCode should adapt imports automatically. This can only be done by the user.

#### P2.14 — panel_app.py monolith split
- **Status:** [ ]
- **Dependencies:** P2.13 (runtime split — interface imports from runtime are stable)
- **Key files:** `vivarium/interface/panel_app.py`
- **CLAUDE.md updates:** `vivarium/interface/CLAUDE.md` — update structure and file table to reflect new modules.

Extract `UpdateManager` and `JupyterManager` into separate modules. Rename `WindowManager` → `PanelApp`.

**`UpdateManager`** (~340 LOC) → `vivarium/interface/update_manager.py` — self-contained: owns its widgets, background thread, calls `updater.py` functions. No dependency on the main app beyond being placed in the layout.

**`JupyterManager`** (~310 LOC) → `vivarium/interface/jupyter_manager.py` — needs `scene_config` (notebook path/port) and `controller` (notebooks dir) passed as constructor args.

**`PanelApp`** (~760 LOC) stays in `panel_app.py` — scene selection, Bokeh plot, controls, streaming, component tabs, callbacks. These share state heavily and splitting further would require an awkward shared-state object.

**Naming rationale:** `PanelApp` rather than `VivariumApp` or `VivariumInterface` — avoids stutter with `vivarium.interface.VivariumInterface`, and distinguishes from component `Interface` subclasses (`BraitenbergInterface`, etc.) which are parts that plug into the app.

**Resulting structure:**
```
vivarium/interface/
  panel_app.py          # PanelApp — core simulation UI
  update_manager.py     # UpdateManager — update checking/download
  jupyter_manager.py    # JupyterManager — Jupyter lifecycle/UI
  parameterized.py      # (unchanged)
  utils.py              # (unchanged)
```

#### P2.15 — Transmit controller_cls via controller_parameters
- **Status:** [ ]
- **Dependencies:** P2.11 (dynamic dataclass fixes — controller_parameters build path is clean)
- **Key files:** `vivarium/simulator/simulator.py`, `vivarium/controllers/vivarium_controller.py`
- **CLAUDE.md updates:** `vivarium/simulator/CLAUDE.md` — update controller parameters section. `vivarium/controllers/CLAUDE.md` — document that controllers can be discovered from controller_parameters without config file.

Enable config-free client discovery. When a researcher builds a Simulator programmatically (no YAML config) and starts a gRPC server, clients (`VivariumController`) currently reload the scene config from disk. If no YAML exists, the client can't discover which controller classes to instantiate.

**Changes:**
- Store `controller_cls` (and `interface_cls`) at the component level in `controller_parameters` — alongside but separate from `controller_kwargs`. Example: `controller_parameters.agents.controller_cls`.
- In `Simulator.from_config()`, extract `controller_cls` from the `client:` block when building `controller_parameters` (currently only `controller_kwargs` is extracted).
- Client reads `controller_parameters.<component>.controller_cls` to discover which class to instantiate — no config file needed.
- No new proto message or RPC required — `controller_parameters` is already transmitted via `GetControllerParameters`.
- If the user wants to launch a server from an existing `Simulator` instance (built without `from_config`), they must fill the relevant `controller_cls` fields in `controller_parameters` before starting the server.

#### P2.16 — Config experiment spike
- **Status:** [ ]
- **Dependencies:** P2.10 (component move — `_target_` strings updated), P2.07 (rigid body removal — simpler configs), P2.11 (dynamic dataclass — controller_parameters build path clean)
- **Key files:** `conf/scene/braitenberg.yaml`, `conf/scene/braitenberg_defaults.yaml`, `vivarium/utils/scene_configs.py`, `scripts/dev/config_free_braitenberg.py`
- **CLAUDE.md updates:** `conf/CLAUDE.md` — document findings and any new resolver patterns.

**This is a decision gate.** Investigate whether configs can be modernized to better align with `__init__` signatures. The outcome determines whether config restructuring enters Phase 2 as additional tasks or stays as a documented future direction.

**Scope:** Single scene (`braitenberg.yaml`) as a spike.

**Investigate:**
- Try restructuring `braitenberg.yaml` so that config structure mirrors `__init__` signatures more closely.
- Investigate OmegaConf custom resolvers to replace Python-side template directives (`_all_values_`, `_range_`, `_random_`; `by_indices` is harder). Currently only a `range` resolver exists in `scene_configs.py`. Reference: https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
- Investigate whether Hydra Structured Configs can help. Reference: https://hydra.cc/docs/tutorials/structured_config/intro/
- It's OK to keep `get_class().from_config()` rather than switching to `hydra.utils.instantiate()` — the goal is structural alignment between configs and `__init__`, not adopting a specific Hydra API.
- `scripts/dev/config_free_braitenberg.py` serves as a reference for what `__init__` signatures actually expect.

**Outcome:** A written assessment of feasibility and effort. If viable and bounded, add config restructuring tasks to Phase 2. If not, document the current conventions in Phase 3 and defer restructuring.

#### P2.17 — Docstring format standardization
- **Status:** [ ]
- **Dependencies:** P2.14 (panel_app split — all code restructuring done)
- **Key files:** all `.py` files with existing docstrings
- **CLAUDE.md updates:** none

Mass-convert existing reST/Sphinx-style docstrings to Google-style format. The codebase has ~150 Google-style and ~172 reST/Sphinx-style docstrings.

Use automated tools (`pyment` or `docconvert`) for the bulk conversion, then review the output for correctness. Focus on converting format, not rewriting content — don't change what the docstrings say, only how they're formatted.

This standardizes the format before Phase 3, where new docstrings will be added to public classes.

---

### Phase 3 — Documentation & Tutorials

#### P3.00 — Finalize journey naming
- **Status:** [ ]
- **Dependencies:** none
- **Key files:** none (decision task)
- **CLAUDE.md updates:** `CLAUDE.md` (global) — update journey references once names are decided.

Decide final names for the four audience journeys before writing any tutorials or the README. Current working names by mode:
- Programmatic JAX (was "Researcher")
- Notebook control (was "CS student")
- Interactive UI (was "Non-programmer / Web interface")
- Component development (was "Developer")

Also decide whether practical session students (interdisciplinary Master students using PyInstaller bundle + sessions 1-4) are a separate 5th journey ("Practical sessions") or a branch within "Notebook control" (two entry paths: pip install → main tutorial, vs PyInstaller bundle → sessions).

Output: agreed names recorded in PLAN.md Key Decisions table.

Subsequent Phase 3 tasks use working names (e.g. "researcher tutorial", "main CS tutorial") and placeholder filenames. Once journey names are finalized, update all Phase 3 task titles, descriptions, and filenames to use the agreed terminology.

#### P3.01 — Researcher tutorial notebook
- **Status:** [ ]
- **Dependencies:** P3.00 (journey names), P2.17 (all Phase 2 done)
- **Key files:** `notebooks/server_side/` (existing, to reference), `scripts/dev/config_free_braitenberg.py` (reference for `__init__`-only path), `vivarium/environment/environment.py`, `vivarium/simulator/simulator.py`, `vivarium/environment/render.py`
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — add new notebook to file table.

Write a new notebook (e.g. `notebooks/tutorials/researcher_tutorial.ipynb` — name TBD per P3.00) for researchers who want to run headless simulations in JAX. Cover:
1. Config loading with `load_scene_config()` and `Environment.from_config()`
2. `Environment` vs `Simulator` — when to use which
3. Stepping in a pure JAX loop, extracting data from state arrays
4. Visualization with `render.py`
5. The `__init__`-only path (no YAML, no Hydra) — reference `scripts/dev/config_free_braitenberg.py`
6. Installing with GPU support (cuda-related extras in `setup.py`)

Mark reproduction as "functional but lightly tested" where relevant.

#### P3.02 — YAML scene creation tutorial
- **Status:** [ ]
- **Dependencies:** P3.01 (researcher tutorial — builds on it), P2.16 (config experiment outcome known)
- **Key files:** `conf/scene/` (YAML files), `conf/CLAUDE.md`, `vivarium/utils/scene_configs.py`
- **CLAUDE.md updates:** `conf/CLAUDE.md` — reference new tutorial.

Write a separate tutorial (notebook or markdown — name TBD per P3.00) for creating custom scenes in YAML. Distinct from the researcher tutorial — this covers the config system specifically. Content depends on the P2.16 config experiment outcome (whether configs were restructured or kept as-is).

Cover: defaults composition and inheritance chain, `_target_` class wiring, `client:` block conventions, template directives (`_all_values_`, `_range_`, `_random_`, `by_indices`), how to add a new scene, testing a scene config.

#### P3.03 — Archive server-side notebooks
- **Status:** [ ]
- **Dependencies:** P3.01 (researcher tutorial written — replacement exists)
- **Key files:** `notebooks/server_side/` (all files)
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — update file table (notebooks archived or removed).

All server-side notebooks use obsolete import paths (`vivarium.environments.braitenberg.simple.simple_env`). Now that a researcher tutorial exists (P3.01), these are no longer needed. Archive or delete them. Delete `notebooks/server_side/README.md` (empty).

#### P3.04 — Main CS tutorial notebook
- **Status:** [ ]
- **Dependencies:** P3.00 (journey names), P2.02 (feature inventory recorded in `tests/controllers/test_edu_sessions/`), P2.17 (all Phase 2 done)
- **Key files:** `tests/controllers/test_edu_sessions/` (feature inventory), `notebooks/sessions/miniproject_template.ipynb` (starting basis), `notebooks/sessions/session_1.ipynb` through `session_4.ipynb`
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — add new notebook to file table.

Write a new main tutorial notebook (e.g. `notebooks/tutorials/main_tutorial.ipynb` — name TBD per P3.00) for CS students comfortable with standard Python. Covers all key concepts from sessions 1-4 and the miniproject in compact form — behaviors as functions, selective sensing, routines, logging, custom configs, etc. — without the verbose explanations needed for non-programmers.

Starting basis: the miniproject notebook's Recap section + features introduced in the miniproject itself. Use the feature inventory from P2.02 (recorded in `tests/controllers/test_edu_sessions/conftest.py`) to ensure completeness.

Document the following concepts within the tutorial:
- Implicit step routing: how multiple controllers share a single simulation loop, which client triggers the step, and how changes propagate.
- `BehaviorHandler.behave()` coupling: behaviors set motor values on agents, `BehaviorHandler` is only used with `AgentController` — this is intentional, not a bug.

#### P3.05 — Revise web interface tutorial
- **Status:** [ ]
- **Dependencies:** P3.00 (journey names), P2.14 (panel_app split done — UI internals stable)
- **Key files:** `docs/web_interface_tutorial.md`
- **CLAUDE.md updates:** none

Revise `web_interface_tutorial.md` to be a proper standalone guide. Before writing, agree on:
1. What to explain (all UI features: scene selection, entity interaction, config tabs, Jupyter embedding, etc.)
2. Structure: explain all parts first then demo a use case, vs. use case as running example
3. A concrete use case: setting up a relatively complex simulation using only the interface (no code)

See also B0.5 below.

The tutorial should demonstrate the concrete use case end-to-end.

#### P3.06 — Developer tutorial notebook
- **Status:** [ ]
- **Dependencies:** P3.00 (journey names), P2.10 (component move done), P2.16 (config experiment outcome known), P2.17 (all Phase 2 done)
- **Key files:** `vivarium/components/` (after P2.10 move), `scripts/dev/config_free_braitenberg.py`
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — add new notebook to file table.

Write a notebook (e.g. `notebooks/tutorials/adding_a_component.ipynb` — name TBD per P3.00) for developers who want to extend Vivarium. Cover:
1. The three-file pattern (`component.py`, `controller.py`, `interface.py`) and which files are optional
2. Component lifecycle: `update_state_cls` → `init_base_entity` → `init_state_fn` → `get_step_function`
3. The `__init__`-only path — instantiate the component, add to Environment, step, verify behavior
4. YAML config integration as an extra — `_target_` wiring, `client:` block, defaults composition, testing

Before writing, decide on a concrete example: reimplementing an existing component (simpler, verifiable) vs. implementing a new one (more motivating).

Document the following within the tutorial:
- Component naming: a component is defined by its `Component` subclass (JAX step function), optionally with a `Controller` and `Interface`. Simpler components have only the `Component` class.

This should be one of the last Phase 3 tasks — requires stable component architecture.

#### P3.07 — Revise main README
- **Status:** [ ]
- **Dependencies:** P3.00 (journey names), P3.01 through P3.06 (tutorials exist to link to)
- **Key files:** `README.md`
- **CLAUDE.md updates:** none

Revise the main README to structure around the audience journeys. Structure: project description → the journeys with links to their respective tutorials → installation → development setup. Use the finalized journey names from P3.00.

#### P3.08 — Update notebook READMEs and relocate PyInstaller docs
- **Status:** [ ]
- **Dependencies:** P3.03 (server-side notebooks archived), P3.04 (main tutorial written)
- **Key files:** `notebooks/README.md`, `notebooks/tutorials/README.md`, `notebooks/server_side/README.md`, `notebooks/sessions/README.md`
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — update README references.

- **Update** `notebooks/README.md` to reflect current notebook state after all changes.
- **Delete** `notebooks/tutorials/README.md` (outdated, replaced by actual tutorials).
- **Keep** `notebooks/sessions/README.md` as-is (serves PyInstaller users).
- **Relocate PyInstaller binary install docs** to a standalone document (e.g. `docs/install_binary.md` or a section in the main README). This is the recommended install path for both the web interface journey and the practical sessions. Currently buried in `notebooks/sessions/README.md`.

#### P3.09 — Fix Google Colab notebook
- **Status:** [ ]
- **Dependencies:** none
- **Key files:** `notebooks/tutorials/google_colab.ipynb`, `notebooks/tutorials/google_colab.md`
- **CLAUDE.md updates:** `notebooks/CLAUDE.md` — update status.

Fix the Google Colab notebook: resolve hardcoded branch reference and address TODOs. Decide on final location (keep in `notebooks/tutorials/` or move).

#### P3.10 — Add docstrings and type hints to public classes
- **Status:** [ ]
- **Dependencies:** P2.17 (docstring format standardized)
- **Key files:** `vivarium/controllers/vivarium_controller.py`, `vivarium/controllers/handlers.py` (after P2.06 rename), `vivarium/interface/panel_app.py`, `vivarium/interface/parameterized.py`, `vivarium/environment/environment.py`, `vivarium/environment/components/component.py`, `vivarium/simulator/simulator.py`
- **CLAUDE.md updates:** none

Add Google-style docstrings and type hints to main public classes and methods. Priority: core abstractions first (`Environment`, `Component`, `Simulator`, `VivariumController`), then student-facing APIs (`Controller.to_deal_with()`, `Controller.remote_to_ctrl()`, `PanelApp.__init__()`, `create_interfaces()`, `ParameterizedData` methods). Internal helpers and internal code deferred post-release.

**Note:** Key file paths listed above reflect the pre-Phase-2 layout. After P2.10 (component move), `vivarium/environment/components/component.py` becomes `vivarium/components/component.py`. Verify actual paths at execution time.

---

### Phase 4 — Feature Fixes & Additions _(time permitting)_

#### P4.00 — Server-side recording component
- **Status:** [ ]
- **Dependencies:** P2.06 (broken recording code removed), all Phase 2 complete
- **Key files:** `vivarium/components/` (new component to create), `vivarium/simulator/simulator.py`
- **CLAUDE.md updates:** `CLAUDE.md` (global) — add recording component to component table.

The broken recording code (`record`, `start_recording`, `stop_recording`, `save_records`, `load`) was removed in P2.06. Design and implement a proper replacement as a server-side JAX component that records simulation data into a dedicated state field.

This serves both the "recording" need (state capture over time for analysis/replay) and the "headless logging" need — researchers running headless simulations get their data from the JAX state rather than needing a client-side logger.

Requires a dedicated design discussion before implementation — decide what to record, storage format (ring buffer vs growing array), memory management, and API for starting/stopping/extracting recordings. Client-side `Logger` (in controllers) stays as-is for the notebook/programmatic use case.

#### P4.01 — Render from recorded data
- **Status:** [ ]
- **Dependencies:** P4.00 (recording component — defines what data is available)
- **Key files:** `vivarium/environment/render.py`
- **CLAUDE.md updates:** none

Add the ability to produce visualizations (frames, videos) from recorded simulation data a posteriori. This depends on what the recording component (P4.00) stores, so rendering-from-recording should be designed alongside recording.

`render.py` stays as a quick matplotlib tool for visual checks. This task adds a proper pipeline for producing videos from headless simulation recordings.

#### P4.02 — Replay recorded simulations in Panel UI
- **Status:** [ ]
- **Dependencies:** P4.00 (recording component — defines storage format and data available)
- **Key files:** `vivarium/interface/panel_app.py`, `vivarium/components/` (recording component from P4.00)
- **CLAUDE.md updates:** `vivarium/interface/CLAUDE.md` — document replay feature.

Add the ability to load a previously recorded simulation and replay it within the Panel UI. This includes loading recorded data, playing back timesteps in the Bokeh visualization, and providing playback controls (play, pause, scrub through timesteps, playback speed). The storage format and available data depend on the recording component designed in P4.00.

#### P4.03 — Braitenberg sensing/motor extensions
- **Status:** [ ]
- **Dependencies:** all Phase 2 complete
- **Key files:** `vivarium/components/braitenberg/sensorimotor.py` (after P2.10 move)
- **CLAUDE.md updates:** `vivarium/environment/CLAUDE.md` — update braitenberg component description.

Two concrete extensions, time permitting:

1. **N evenly-spaced sensor cones:** Generalize proximeters from 2 fixed sensors (left/right) to an arbitrary number of regularly spaced sensor cones. Touches `sensorimotor.py` (sensor geometry, motor mapping) and likely the behavior system (behavior functions currently assume 2 inputs). Bounded but non-trivial.

2. **Non-occluding proximeters:** Currently, a proximeter senses the closest entity in its cone regardless of subtype — non-target entities can shadow target entities. Add a non-occluding mode where the proximeter senses the closest entity of the target subtype, ignoring non-target entities in the way.

The current 2-proximeter occluding setup works and is well-exercised by sessions 1-4 and tests. No audience journey is blocked by this.

---

### Phase 5 — Release Prep

- Full test run on a clean environment
- Version bump and changelog
- GitHub release

---

## Key Decisions

| Decision | Choice |
|---|---|
| SceneBuilder | Removed — not in release scope |
| Multi-spawn | Implemented and tested — keep |
| Target audiences | Researchers, CS students, younger students (web) |
| Release standard | Clean and usable, not exhaustive |
| Phase structure | Phase 2 (cleanup/refactoring) → Phase 3 (docs/tutorials) → Phase 4 (feature fixes, time-permitting) |
| CLAUDE.md role | Descriptive only — diagnostic content stripped and moved to task descriptions |
| Docstring format | Google-style (mass-convert existing reST in P2.17, new docstrings in P3.10) |
| Journey naming | By mode (Programmatic JAX, Notebook control, Interactive UI, Component development) — finalized in Phase 3 |
| Config experiment | Spike on braitenberg.yaml in Phase 2 — outcome may add tasks |

## Deferred (post-release)

_Items considered during the audit but explicitly deferred. Full context preserved for future reference._

**D.01 — Missing `__all__` exports.**
No package defines `__all__`. This means `from vivarium.controllers import *` exports everything, and IDE autocompletion shows internal symbols. Adding `__all__` to each package's `__init__.py` would make the public API explicit. No impact on functionality or user journeys — purely a code quality improvement.

**D.02 — Repetitive controller/interface code.**
Each entity controller (`EntityWrapper`, `EntityController`, `AgentController`, `WallController`) maps user-facing property names to state paths via `__getattr__`/`__setattr__`. The routing logic branches on whether the field is in `entity_state`, a component-specific field, a controller parameter, or a split attribute (e.g. `left_motor` → `motor[0]`). This logic is spread across four classes with significant duplication. `AttributeMapping` exists as a helper but doesn't eliminate the routing complexity. A declarative field registry could reduce duplication, but designing it properly (handling all branch cases) is non-trivial and risks regressions. No user is affected by the current verbosity.

**D.03 — RoutineHandler/BehaviorHandler base class extraction.**
`RoutineHandler` and `BehaviorHandler` share ~80% of their code (registry pattern: attach/detach/start/stop/print, two-dict pattern, threading lock, name resolution). Only `behave()` (weighted motor blending) and `routine_step()` (direct call + error auto-removal) differ. A base class would be clean but is low priority — internal code, works correctly, no divergence risk. The rename from `utils.py` → `handlers.py` is done in P2.06.

**D.04 — Platform PID functions visibility.**
`handle_server_interface.py` (or its post-split successors in `vivarium/runtime/`) contains platform-specific PID lookup and process management functions with underscore-prefixed names. These are purely internal — no external consumers. Visibility cleanup is cosmetic.

**D.05 — Hardcoded scene enumeration.**
`get_available_scenes()` in `scene_configs.py` categorizes scenes by name pattern (`startswith('session')`, etc.) with `research` as the catch-all default. Adding a new category or moving a scene requires editing the function. Brittle but functional for the current stable scene set.

**D.06 — Implicit client inclusion.**
Component `client:` blocks in YAML configs are not visible when reading a scene file that inherits from `braitenberg_defaults.yaml` — they're defined in the defaults chain. A developer adding a new component might not realize they need a `client:` block. The developer tutorial (P3.06) documents this inheritance chain, which is the more impactful fix. A code-level solution (e.g. explicit client registration) is deferred.

**D.07 — Global state in `run_interface.py`.**
`run_interface.py` uses module-level globals for process handles and cleanup state. Acceptable for a script entry point but not ideal. Refactoring to a class would improve testability.

**D.08 — Notebook execution order enforcement.**
Sessions 1-4 are numbered and reference prior sessions when building on earlier concepts, but nothing prevents a student from running them out of order. Code-level enforcement (e.g. checking session completion markers) would be fragile and over-engineered. The tutorial progression is documented in P3.04.

**D.09 — `--version` flag for CLI scripts.**
`get_version()` exists in `runtime.py` (or `runtime/paths.py` after P2.13). Adding a `--version` flag to `run_server.py` and `run_interface.py` is trivial but not needed for any audience journey.

**D.10 — Checksum validation for PyInstaller updates.**
The updater downloads new PyInstaller binaries but doesn't verify checksums. Requires server-side changes (hosting checksums alongside binaries) in addition to client-side validation.

**D.11 — MaskFunction extensibility.**
`MaskFunction` in the Braitenberg component only supports one label (`'exists'`). Making it extensible is a design question tied to future use cases. If P4.03 extensions (N sensors, non-occluding proximeters) need richer masking, that's the right time to generalize.

**D.12 — Config validation.**
No validation that `subtype_labels` in entity configs match actual subtypes used in behavior/sensing configs. Developer-facing guardrail — all current scenes have correct labels. The developer tutorial (P3.06) documents the requirement.

**D.13 — Thread safety proper fix.**
P2.03 adds explanatory comments about the threading risk in the Panel update loop (`update_plot_cb()` reads state with no locking while gRPC callbacks update it from another thread). CPython's GIL makes torn reads unlikely, but `state` and `controller_parameters` are updated in two separate assignments — the callback could read between them. A proper fix (locking or atomic state+params swap) risks introducing deadlocks. Deferred until the Panel UI gets deeper investment.

**D.14 — `StateAndControllerParameters` dual definition.**
`StateAndControllerParameters` is defined in both `simulator.py` and `grpc_server/simulator_client.py` with slightly different structures. Low risk since they serve different contexts (server vs client), but could be unified.

**D.15 — Dynamic Param field cleanup in `interface/utils.py`.**
`interface/utils.py` dynamically creates `param.Parameter` fields from `controller_parameters`. If entity types change between scenes, stale fields from a previous scene could persist. Low risk given the current single-scene-per-session usage, but would matter if scene switching becomes dynamic.

**D.16 — Test coverage gaps.**
Modules with no direct unit tests (covered indirectly by integration tests): physics components (collision, friction, reset), sensorimotor, consumption, walls, ProximityMap, Particle Lenia, controller internals (Logger, RoutineHandler, BehaviorHandler), Simulator internals (SimulatorController, streaming rate limiting), Panel UI beyond smoke test, parts of utils. After P2.14 (panel_app split), `UpdateManager` and `JupyterManager` become self-contained and testable.

**D.17 — Test structural improvements.**
Stub files for untested modules (makes gaps visible in file tree). Directory-level test markers as an alternative to per-file markers.

**D.18 — Standalone API reference documentation.**
A reference document or generated docs (Sphinx) covering the full public API. Tutorials (P3.01–P3.06) + docstrings (P3.10) are sufficient for the release; a standalone reference is a post-release improvement.

## Discovered Bugs

_Bugs found during task execution that are outside the scope of the current task. Fix these when working on the relevant area, or as a standalone fix if blocking._

#### B0.2 - Some tests marked as `@pytest.mark.slow` (i.e. using subprocess and/or grpc) might be as useful using an in-process `Simulator` instance
For instance `test_proximeters_selective_filters_by_subtype` or `TestSpawnController` (but to see). More generally, we should decide for general rules enabling to decide if a test should be "fast" or "slow", then check if every existing test are consistent with these rules. For now, the decision looks relatively arbitrary. We both want to minimize the amount of slow tests, yet use grpc whenever it is relevant for a giving test. Need a discussion on what the rules should be. 
Another related discussion to have: instead of `@pytest.mark.slow` (or in addition to), shall we have a ``@pytest.mark.grpc` for tests that involve a client-server grpc connection? This could be useful when we make changes that can only break under this use case. But we might also need a marker for a client-server setup with in-process simulator.

#### B0.4 - Check the logic of Environment.step()
In particular that both `self.neighbor_manager.reallocate_if_overflow` and `neighbors.did_buffer_overflow` are called (only in debug mode though, so maybe not a big deal)

#### B0.5 - Revise the `demo.yaml` scene
Potential options:
- A scene that serves to demonstrate how to configure a relatively large prey-predator simulation from the Panel UI (could be useful for the Web Interface journey).
- A scene that directly configures a large prey-predator simulation (no need to configure anything from the interface or notebook, just play it)

---

## Completed
_In chronological order_

#### P2.00 — Strip CLAUDE.md files to descriptive content
Removed diagnostic sections (Known Issues, Refactoring Opportunities, Structural Questions, Cross-Package Issues, Dead Code to Remove, Audience Journey Readiness) from all 10 CLAUDE.md files. Removed LOC/Size columns, audit-related titles, and fixed dangling references.

#### P2.01 — Fix test_reproduction and add reproduction tests
The reproduction fixture's consumption disablement (`range=0`) was insufficient — overlapping entities still triggered consumption. Changed to `start=False`. Added 3 targeted reproduction tests (birth, recovery time gating, non-existing entity exclusion). Full pytest green (191 passed).

#### P2.02 — Write exhaustive feature tests for student-facing API
Replaced `tests/test_edu_sessions.py` with `tests/test_edu_sessions/` package (9 test files + conftest). Complete feature inventory (13 categories) in conftest docstring. 101 tests total: 72 new covering entity access, properties, sensing, behaviors, routines, logger, consumption/spawn, internal state + 30 moved subtype_labels tests - 1 dropped (superseded). Discovered B.01 (SingleSpawnController `.item()` bug, 3 xfail tests). Updated P2.05 target structure. Full pytest green (262 passed, 3 xfailed).

#### B.01 — Fix `SingleSpawnController.__getattr__` crash on in-process path
`SingleSpawnController.__getattr__` assumed values were always numpy arrays (`.item()`, `.tolist()`), but the in-process Simulator returns JAX arrays or plain Python types wrapped in `Remote` proxies. Fixed by unwrapping `Remote` and normalizing to numpy via `np.asarray()` before calling array methods. Removed 3 xfail markers from `test_consumption_spawn.py`. Full pytest green (262 passed, 0 xfailed).

#### P2.03 — Early cleanup batch (workflow shakedown)
Deleted 4 redundant/outdated files (`scripts/run_vivarium.py`, `session_5_logging.ipynb`, `session_6_bonus.ipynb`, `quickstart_tutorial.ipynb`). Removed misleading TODO on `apply_changes()`, removed explicit `scene_name` setter (Python's default `AttributeError` suffices) and added a read-only test. Added `notebooks/sandbox.ipynb` to `.gitignore`. Added threading note to `update_plot_cb()` docstring in `panel_app.py`.

#### P2.04 — Test suite quality fixes
Added assertions to no-op tests (`test_instantiate`, `test_braitenberg`). Renamed shadowed duplicate `test_remote` → `test_remote_fetch_and_update` / `test_remote_apply`. Uncommented and fixed `test_is_connected_verify_detects_no_server` with mock gRPC client. Removed `# assert False` in `test_panel_app.py`. Replaced silent NaN revert in `test_environments.py` with assertion. Added `boyds` to scene parametrize. Registered `slow` marker in `.pytest.ini` and added `@pytest.mark.slow` to individual slow tests across 12 files. Added docstrings to 19 fixtures in `conftest.py`. Fixed `test_no_behavior_no_motion` → `test_no_behavior_zero_motors` (checks motors instead of position). Added motor assertions to `test_behavior_produces_motion`. Full pytest green.

#### B0.3 — Fix routine test assertions and underlying VivariumController bugs
Investigation of loose routine test assertions (`>= 3` instead of `== 3`, getting 5) revealed three bugs in `VivariumController`: (1) `start_session` started an unintended background controller thread by not passing `start_controller_thread=False` to the constructor, (2) `step()` didn't increment `self.time`, breaking interval-based routines when stepping manually, (3) `start_session` returned with stale local state (`simulation_running`, `run_from`) because deferred changes weren't flushed. Fixed all three and tightened test assertions to exact counts. Full pytest green (262 passed).

#### P2.05 — Test directory restructuring
Restructured `tests/` to mirror source package structure: `environment/`, `simulator/`, `controllers/`, `interface/`, `utils/`, `scripts/`. Split root `conftest.py` — component fixture chain moved to `environment/conftest.py`. Renamed files: `test_environments.py` → `test_environment.py`, `test_update_check.py` → `test_updater.py`, `test_dataclass_api.py` → `test_dataclass_wrapper.py`. Moved `test_edu_sessions/` under `controllers/`. Removed dead `_make_state` method from `test_multi_spawn.py` (contained unused `from conftest import remove_duplicates`). Added test directory mirroring guideline to General Guidelines. Full pytest green (271 passed).

#### P2.06 — Dead code removal
Deleted 2 files (`eco_evo/component.py` empty, `session_5_logging copy.ipynb` obsolete). Fixed `scripts/profiling.py` (broken `SceneConfiguration` import → current `load_scene_config` API). Removed recording feature from `simulator.py` (`start_recording`, `record`, `save_records`, `stop_recording`, `load`, plus `save`/`saving_name` params from `run()`/`_run()`). Removed `nested_fields_to_access` dicts from `simulator.py` and `environment.py`, commented decorator from `simulator_client.py`. Removed `SetState` handler from `simulator_server.py` (called non-existent method; proto definition was already gone). Removed `notebook_mode` from `panel_app.py`. Removed `kill_session()` from `controllers/utils.py` and `set_nested_attr` from `controllers/__init__.py` exports. Renamed `controllers/utils.py` → `controllers/handlers.py` (updated 3 import sites). Cleaned up unused imports (`proto_to_ndarray` in server, `Logger` in braitenberg controller). Simplified `tests/CLAUDE.md` coverage section (removed stale counts, corrected Logger test status). Updated CLAUDE.md files for simulator, controllers, scripts, and tests. Full pytest green (271 passed).

#### P2.07 — Rigid body removal
Removed all rigid body support across 19 source/test files. No scene used rigid bodies; the conditional branching added complexity throughout. Deleted: `to_rigid_body_state()`, `is_rigid_body()`, `unified_*` accessors, `BaseEntityState.__getattr__`, `handle_rigid_body` decorator, `to_rigid_body()`, `create_property()` + 8 class-level RigidBody properties in EntityWrapper, `rigid_body_to_point_particle()`, `RigidBody` proto message. Simplified `BaseState.__getattr__`, `sum_forces()`, `mask_momentum()`, `motor_force()`, `sum_force_to_entities()`, and force accumulation in collision/friction/reset/step components. Replaced `unified_*` → direct field access in environment.py, sensorimotor.py, friction, step. Replaced `position_center` → `position` and `position_orientation` → `orientation` in render.py and 5 test files. Regenerated proto files. Removed rigid body fixture/parametrization from test_dataclass_wrapper.py. Updated environment/CLAUDE.md. 60 targeted tests pass.

#### P2.08 — Change `exists` field from int to boolean
Changed `entity_state.exists` from int (0/1) to boolean across 12 files. Core: `dtype=int` → `dtype=bool` in environment.py. Simplified comparisons: `exists == 1` → `exists`, `exists == 0` → `~exists` in environment.py, utils.py, consumption, render.py. Fixed assignments: `.set(0)` → `.set(False)`, `.set(1)` → `.set(True)` in eco_evo/utils.py and reproduction. Removed `exists` entry from `get_entity_parameter_mapping` (default mapping handles bool natively). Removed redundant `bool()` wrappers in controller.py (print_infos), component.py (to_config), interface.py. Updated 4 test files. Full codebase audit confirmed no arithmetic patterns (`exists * value`) — only comparisons, masks, and `.sum()` (which works correctly with booleans). Full pytest green (271 passed).

#### P2.09 — Bug fixes
Fixed bugs across 10 source files + 12 config files. **render.py**: removed non-idiomatic `jnp.where(exists)` pattern (boolean mask suffices), fixed double `[exists]` indexing on diameter/orientation, fixed `plt.xlim` → `plt.ylim`, replaced fragile color count with `len(pos)`, removed unused `jax.numpy` import. **Typos/smells**: renamed `udpate_other_interfaces` → `update_other_interfaces` (interface.py + panel_app.py), `except:` → `except Exception:` (handle_server_interface.py), `lg.error` → `lg.exception` in `_start_server_cb` (panel_app.py). **Config**: added `- _self_` to defaults in all 12 scene configs that were missing it. **Simulator**: replaced hardcoded `../../conf/scene/simulator` path in `to_config()` with `runtime.get_config_dir()` + `hydra.initialize_config_dir()`. Added `grpc_error_handler` decorator to all 13 unary RPC handlers and try/except with `context.abort(INTERNAL)` in both streaming RPCs (simulator_server.py). Added `test_grpc_error_handler` test. Full pytest green.

#### P2.10 — Component package move
Moved `vivarium/environment/components/` → `vivarium/components/` (top-level package). Updated all Python imports (~40 files in components/, tests/, scripts/, interface/), all `_target_` and `*_cls` strings in YAML configs (~20 references), and `vivarium_multi.spec`. Removed re-export shim directories (`vivarium/controllers/components/` and `vivarium/interface/components/` — 8 `__init__.py` files, no longer needed). Fixed deep relative imports that broke after the move (5 files in components/ that used `....` / `.....` to reach `controllers/` — converted to absolute imports). Fixed `vivarium/environment/__init__.py` stale `from vivarium.environment import components`. Updated CLAUDE.md files (global, environment, controllers, interface, tests ; new components/CLAUDE.md). Full pytest green (272 passed).