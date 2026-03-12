# Vivarium

Multi-agent simulation framework built on JAX. Entities (Braitenberg vehicles, objects, Particle Lenia, walls) interact in a 2D physics environment. Three modes of use:

1. **Headless JAX simulation** — researchers run large-scale simulations on GPU
2. **Programmatic Python control** — CS students control agents from notebooks via `VivariumController`
3. **Web interface** — younger students interact via a Panel/Bokeh UI with no coding

## Tech Stack

JAX 0.8.2 + JAX-MD 0.2.27 (physics), gRPC 1.71.2 (client-server), Panel 1.8.5 + Bokeh (web UI), Hydra 1.3.2 + OmegaConf (config), Pytest. Python 3.10-3.12.

---

## Architecture

### Package Dependency Map

```
vivarium/utils/          ← foundation (no vivarium deps)
  ↑
vivarium/environment/    ← JAX simulation core (depends on utils)
  ↑
vivarium/simulator/      ← state management, gRPC bridge (depends on environment, utils)
  ↑
vivarium/controllers/    ← user-facing Python API (depends on simulator, environment, utils)
  ↑
vivarium/interface/      ← Panel web UI (depends on controllers, simulator, environment, utils)
  ↑
scripts/                 ← entry points (depends on all packages)
```

**No circular imports**, but a **design flaw** in the dependency directions: component controllers and interfaces are defined inside `vivarium/environment/components/` yet they are client-side code that imports base classes from `vivarium/controllers/` and `vivarium/interface/`. This means environment — which should be a pure simulation engine — depends upward on packages that are supposed to abstract over it. See "Cross-Package Issues" below and controllers/CLAUDE.md for analysis and a proposed fix.

### Data Flow

```
Client (notebook/Panel)                    Server
─────────────────────                    ──────
VivariumController                       Simulator
  ├── controllers = {name: Controller}     ├── env (Environment)
  ├── client (SimulatorGRPCClient)         │     ├── components (JAX step fns)
  │     ├── state (numpy)     ←── gRPC ──  │     └── state (JAX arrays)
  │     ├── controller_params ←── gRPC ──  ├── controller_parameters (dynamic dataclass)
  │     └── step(changes)     ─── gRPC ──→ └── step() → env.step(state)
  └── Remote proxy (collects changes)
```

**Two data channels:**
- **State** (JAX arrays): simulation state (positions, forces, energy, etc.). Server holds JAX; client gets numpy copies via gRPC. Full state transferred each time (no deltas).
- **Controller parameters** (non-JAX): configuration data (freq, scene_name, subtype_labels, per-component settings). Built dynamically at startup from Hydra config via `create_dataclass_from_dict()`. Synchronized centrally by Simulator — clients fetch copies via RPC.

**Change flow:** Controllers manipulate a `Remote` proxy that records `(path, value)` changes. `apply_changes()` batches and sends them to the server via `step(changes)`. Changes can target either state fields or controller parameters — the server applies both.

`Simulator` and `SimulatorGRPCClient` share a duck-typed interface — `VivariumController` accepts either.

### Component Architecture

Each simulation feature is implemented as a `Component` subclass (defined in `component.py`). The base `Component` class defines the server-side lifecycle that `Environment` calls:

1. `from_config(config)` — instantiate from Hydra config
2. `update_state_cls(state_cls)` — add fields to the state class
3. `init_base_entity(entity_state)` — for entity components: populate entity_state arrays
4. `init_state_fn(state, neighbor_manager, key)` — initialize component-specific state fields
5. `get_step_function(state, neighbor_manager, key)` — return a pure JAX step function: `fn(state, neighbors, key) → state`

At runtime, `Environment.step()` calls these step functions in `precedence` order (lower first). Each function receives the full state and returns a new state — pure functional, no side effects, JIT-compatible.

Components live in `vivarium/environment/components/` and follow a three-file pattern:

```
component_name/
  ├── component.py      # Component subclass: JAX step function (server-side, pure functional)
  ├── controller.py     # Client-side Python API (getattr/setattr → state and controller parameters paths)
  └── interface.py      # Panel UI (Bokeh renderer + param controls)
```

Not all components have all three files — simpler ones (reset, friction, energy, reproduction) only have `component.py`.

**Component execution order** (`precedence`, lower runs first):

| Precedence | Component | Purpose |
|------------|-----------|---------|
| 0 | Reset | Clear forces |
| 2 | Collision | Soft sphere overlap resolution |
| 3 | Friction | Velocity damping |
| 20 | Spawn | Generate new entities |
| 40 | Consumption | Energy transfer via proximity |
| 60 | Energy | Decay/burst/clamp |
| 80 | Reproduction | Birth/death by energy thresholds |
| 1000 | Step | Verlet integration (always last) |

### State System

`create_state_cls()` dynamically builds the state class:
1. Starts from `BaseState` (time + entity_state)
2. Each component's `update_state_cls()` adds fields
3. Result: JAX-MD dataclass (immutable, JIT-compatible)

`BaseEntityState` is a flat array for all entities: position, momentum, mass, force, orientation, entity_type, entity_type_idx, entity_subtype, exists, diameter, friction.

### Config System (Hydra)

All scenes compose from YAML files in `conf/scene/`:

```
base_scene.yaml
  ↑ braitenberg_defaults.yaml (+ physics pipeline + braitenberg entity + clients)
    ↑ session_defaults.yaml (+ session-specific environment config)
      ↑ session_1, session_2, ...
    ↑ braitenberg, quickstart, sandbox, demo, fishing, non_transitive, boyds
  ↑ (directly) particle_lenia, excretion, lenia_braitenberg
```

`braitenberg_defaults.yaml` is the key anchor — it wires `_target_` classes and `client` configs for components. All Braitenberg-based scenes inherit from it.

### `from_config` Pattern

All major classes use `@classmethod from_config(config)` for Hydra-driven construction. The pattern is consistent but complexity varies:

- **`Component`/`Controller`**: thin wrappers (extract config keys → `__init__`)
- **`EntityComponent`**: generates positions/orientations, expands `_all_values_` templates, converts subtype labels to indices
- **`Environment`**: recursively calls `from_config` on all component factories, creates `NeighborManager`
- **`Simulator`**: extracts controller parameters from all component configs, creates dynamic dataclass, recursively creates `Environment`

All classes can be instantiated via `__init__` without Hydra, but you must replicate the preprocessing that `from_config` does. `hydra.utils.get_class(_target_).from_config(config)` is the universal instantiation pattern.

The config structure does not follow [Hydra's recommended pattern](https://hydra.cc/docs/advanced/instantiate_objects/config_files/) where configs mirror `__init__` signatures. See "Cross-Package Issues → Config–code structure mismatch" for analysis.

### Usage by Mode

**1. Headless JAX simulation** (researchers — no gRPC, no UI):

```python
from vivarium.environment import Environment
from vivarium.utils.scene_configs import load_scene_config

config = load_scene_config("braitenberg")
env = Environment.from_config(config.environment)
state = env.init_state()

for t in range(1000):
    state = env.step(state)  # pure JAX, JIT-compiled
    positions = state.entity_state.position  # JAX array (n_entities, 2)
```

Or via `Simulator` (higher-level wrapper, manages state internally):

```python
from vivarium.simulator import Simulator
from vivarium.utils.scene_configs import load_scene_config

config = load_scene_config("braitenberg")
simulator = Simulator.from_config(config.simulator)

for t in range(1000):
    simulator.step()
state = simulator.get_state()
```

**2. Programmatic Python control** (CS students — gRPC client/server):

```python
from vivarium.controllers import VivariumController

controller = VivariumController(start_server=True, scene_name="braitenberg")
# Spawns gRPC server subprocess → connects SimulatorGRPCClient → builds controllers from config
agent = controller.agents[0]
agent.left_motor = 1.0
controller.step()
print(agent.proximeters())
controller.close()
```

**3. Web interface** (younger students — CLI launch):

```bash
python scripts/run_interface.py  # starts gRPC server + Panel web UI, opens browser with scene selection
```

---

## Local Audit Files

Each package has a detailed `CLAUDE.md` with purpose, API, known issues, test coverage, and refactoring opportunities:

| Package | Audit File | Status |
|---------|-----------|--------|
| `vivarium/environment/` | [CLAUDE.md](vivarium/environment/CLAUDE.md) | Core simulation — solid, ~56 files |
| `vivarium/simulator/` | [CLAUDE.md](vivarium/simulator/CLAUDE.md) | gRPC bridge — solid with dead code |
| `vivarium/controllers/` | [CLAUDE.md](vivarium/controllers/CLAUDE.md) | User API — solid |
| `vivarium/interface/` | [CLAUDE.md](vivarium/interface/CLAUDE.md) | Panel UI — brittle monolith |
| `vivarium/utils/` | [CLAUDE.md](vivarium/utils/CLAUDE.md) | Utilities — solid foundation |
| `conf/` | [CLAUDE.md](conf/CLAUDE.md) | Hydra configs — 48 YAML files |
| `scripts/` | [CLAUDE.md](scripts/CLAUDE.md) | Entry points — mostly solid |
| `tests/` | [CLAUDE.md](tests/CLAUDE.md) | 17 test files, ~98 tests |
| `notebooks/` | [CLAUDE.md](notebooks/CLAUDE.md) | Sessions 1-4 active, rest outdated |

---

## Cross-Package Issues

### Architectural

1. **Component controllers and interfaces live in the wrong package.** They are defined in `environment/components/*/controller.py` and `interface.py`, yet they are client-side code that imports base classes from `controllers/` and `interface/`. This inverts the intended dependency direction: the environment (simulation engine) depends upward on the packages that are supposed to abstract over it. **Proposed fix:** move `vivarium/environment/components/` to `vivarium/components/` as a top-level package. This works because `environment.py` resolves component classes dynamically via Hydra (`hydra.utils.get_class(_target_)`) — no static imports. See [controllers/CLAUDE.md](vivarium/controllers/CLAUDE.md) for full analysis.

2. **Two monoliths need splitting:**
   - `interface/panel_app.py` (1410 LOC) — god-object `WindowManager` handles scene selection, simulation UI, updates, Jupyter, all callbacks. See [interface/CLAUDE.md](vivarium/interface/CLAUDE.md).
   - `utils/handle_server_interface.py` (881 LOC) — mixes process management, server lifecycle, Jupyter, ngrok. See [utils/CLAUDE.md](vivarium/utils/CLAUDE.md).

3. **Dynamic dataclass in Simulator** (`create_dataclass_from_dict('ControllerParameters', ...)`) creates runtime structures with no type safety. Field names come from config, not from code declarations. See [simulator/CLAUDE.md](vivarium/simulator/CLAUDE.md).

4. **RigidBody support is live but dormant.** `state.py`, physics components, and entity controllers all have conditional rigid body paths. Currently no scene uses rigid bodies (all tests assert `not is_rigid_body()`). The code is maintained but untested in rigid body mode. See [environment/CLAUDE.md](vivarium/environment/CLAUDE.md).

5. **Config–code structure mismatch.** Component configs mix constructor args, client-side metadata (`client:` block), and template directives (`_all_values_`, `by_indices`) in the same YAML node. This prevents using Hydra's `instantiate()` and forces every class to have a `from_config` classmethod that filters/transforms before calling `__init__`. The most actionable fix is separating `client` configs from component configs. See [conf/CLAUDE.md](conf/CLAUDE.md) "Config–Code Structure Mismatch" for full analysis and per-class feasibility.

### Patterns to Fix

6. **Typo `udpate_other_interfaces`** appears in both `environment/components/interface.py:65` and `interface/panel_app.py:645`. Same typo, two packages. See [environment/CLAUDE.md](vivarium/environment/CLAUDE.md) and [interface/CLAUDE.md](vivarium/interface/CLAUDE.md).

7. **Bare `except:` in `utils/handle_server_interface.py:469`** — catches everything including KeyboardInterrupt. Should be `except Exception:`. See [utils/CLAUDE.md](vivarium/utils/CLAUDE.md).

8. **State streaming is ineffective in Panel UI.** `_on_state_stream_update()` sets a threading.Event flag, but `update_plot_cb()` never checks it. The periodic callback polls at 33ms regardless. May actually work via a different path (direct state assignment) — needs investigation before removing. See [interface/CLAUDE.md](vivarium/interface/CLAUDE.md).

9. **Recording feature is broken.** `simulator.py` has `start_recording()`/`stop_recording()`/`record()` methods marked "probably broken for now". Dead code in `step()` checks `self.recording`. Either remove or fix. See [simulator/CLAUDE.md](vivarium/simulator/CLAUDE.md).

---

## Dead Code to Remove

### Files

| File | Reason |
|------|--------|
| `vivarium/environment/components/eco_evo/component.py` | Empty file (0 bytes) |
| `scripts/print_config.py` | Unused utility, never invoked |
| `conf/scene/session_6.yaml` | Outdated, no `scene_name`, untested |
| `notebooks/sessions/session_5_logging copy.ipynb` | Duplicate backup (290KB) |

### Code Artifacts

| Location | What | Reason |
|----------|------|--------|
| `environment/components/component.py:13` + `entities/component.py:29` | `is_entity_component` flag | Set but never read |
| `simulator/simulator.py:36-43` | `nested_fields_to_access` dict | Marked "Now unused?" — confirmed dead |
| `simulator/grpc_server/simulator_client.py:24` | Commented-out decorator reference | Related to above dict |
| `simulator/grpc_server/simulator_server.py:112-118` | `SetState` RPC handler | Calls non-existent `simulator.set_state()` |
| `simulator/simulator.py` | Recording feature (`record`, `start_recording`, `stop_recording`, `save_records`, `load`) | Marked broken, untested |
| `interface/panel_app.py:687` | `streaming_toggle` widget | Created but never displayed |
| `interface/panel_app.py:972` | `streaming_toggle_cb()` callback | Defined but never registered |
| `interface/panel_app.py:84` | `self.notebook_mode` | Set but never read |
| `controllers/utils.py` | `kill_session()` | Only used in outdated `session_6_bonus.ipynb` |
| `environment/utils.py:50-91` | `rigid_body_to_point_particle()` | Marked deprecated, only referenced in sandbox notebook |
| `scripts/profiling.py` | Entire file | Uses non-existent `SceneConfiguration` import |
| `simulator/grpc_server/simulator_client.py` | `bidirectional_step_sync()` + related | Implemented but no active code path uses it; `use_streaming` flag is vestigial |
| `conf/scene/interface/base_interface.yaml` | `use_streaming: true` | Read by Panel app but never consumed |

---

## Audience Journey Readiness

### 1. Headless JAX Simulation (Researchers)

**Status: Blocked by documentation.**

- **What works:** `Environment` and `Simulator` APIs are solid. `Simulator.from_config()` creates a working simulation. JAX step functions are JIT-compatible.
- **What's missing:** No current documentation. Server-side notebooks (`notebooks/server_side/`) all use obsolete import paths (`vivarium.environments.braitenberg.simple.simple_env`). No tutorial showing the headless workflow with current API.
- **Blocking:** Need a new tutorial showing: load config → create Environment → step in pure JAX loop → extract data. Or: create Simulator → step without gRPC.

### 2. Programmatic Python Control (CS Students)

**Status: Mostly ready, some gaps.**

- **What works:** Sessions 1-4 are active, pedagogically excellent, and use current API (`VivariumController.start_session()`). `miniproject_template.ipynb` and `reactive_rl.ipynb` are functional. Test coverage is good (`test_edu_sessions.py` — 31 tests).
- **What's incomplete:**
  - Session 5 (logging) marked "still has to be updated"
  - Session 6 (eco-evo) uses deprecated API (`kill_session()`, old patterns) — needs full rewrite
  - `quickstart_tutorial.ipynb` is outdated (uses `NotebookController`)
  - No standalone API reference tutorial (sessions are progressive, not reference)
- **Not blocking but would improve:** A quickstart tutorial rewritten for current API.

### 3. Web Interface (Younger Students)

**Status: Functional but poorly documented and brittle.**

- **What works:** Panel app launches, renders entities, supports drag-drop, component config tabs, start/stop. `web_interface_tutorial.md` exists as a markdown guide.
- **What's brittle:** `WindowManager` is a 1410-line monolith with minimal test coverage (1 test). Update system, Jupyter integration, and streaming are all untested.
- **What's missing:** No end-to-end tutorial beyond the markdown file. No testing of the interactive features.

---

## Test Suite Overview

17 test files, ~98 tests. Mix of unit (~60%) and integration (~40%). Fixture architecture in `conftest.py` supports both in-process gRPC (fast) and subprocess (slow) server modes.

**Well-tested:** VivariumController (42 tests across 2 files), utils/runtime (24 tests), utils/updater (13 tests), gRPC serialization (7 tests), multi-spawn (13 tests), dataclass wrapper (6 tests).

**Not tested:** Individual physics components, braitenberg sensorimotor, entity interfaces, Logger, recording feature, `render.py` (has known bugs), most interactive Panel features.

**Known issues:** `test_reproduction` has a pre-existing failure. Duplicate `test_remote()` function names in `test_dataclass_api.py`. Several no-op tests (`pass` only).

---

## Development

### Running Tests

```bash
pytest                           # all tests
pytest tests/test_edu_sessions.py  # educational session integration tests
pytest -x                        # stop on first failure
```

### Starting a Server (dev mode)

```bash
python scripts/run_server.py scene=braitenberg    # gRPC server only
python scripts/run_interface.py braitenberg         # Panel interface, from which a scene can be selected and the server launched
```

### Programmatic Usage

```python
from vivarium.controllers import VivariumController

controller = VivariumController(start_server=True, scene_name="braitenberg")
agent = controller.agents[0]
agent.left_motor = 1.0
controller.step()
print(agent.proximeters())
controller.close()
```

### Recompiling Protobuf (after editing `.proto`)

```bash
python -m grpc_tools.protoc -I./vivarium/simulator/grpc_server/protos \
  --python_out=./vivarium/simulator/grpc_server/ \
  --pyi_out=./vivarium/simulator/grpc_server/ \
  --grpc_python_out=./vivarium/simulator/grpc_server/ \
  ./vivarium/simulator/grpc_server/protos/simulator.proto
```

### Important Conventions

- **JAX immutability**: All simulation logic must use pure functions. No in-place mutations — return new states.
- **PyInstaller compatibility**: All frozen-mode logic is centralized in `vivarium/utils/runtime.py`. Do not add `sys.frozen` or `sys._MEIPASS` checks elsewhere — use `is_frozen()`, `get_app_root()`, `get_config_dir()` from `runtime.py`.
- **No JAX on client side**: Client code (controllers, interface) uses numpy only. JAX arrays live on the server.
