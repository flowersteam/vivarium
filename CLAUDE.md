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
vivarium/components/     ← component implementations (depends on environment, controllers, utils)
  ↑
vivarium/simulator/      ← state management, gRPC bridge (depends on environment, utils)
  ↑
vivarium/controllers/    ← user-facing Python API (depends on simulator, environment, utils)
  ↑
vivarium/interface/      ← Panel web UI (depends on controllers, simulator, environment, utils)
  ↑
scripts/                 ← entry points (depends on all packages)
```

**No circular imports.** Components live in `vivarium/components/` as a top-level package. Component controllers and interfaces import base classes from `vivarium/controllers/` and `vivarium/interface/`. `environment.py` resolves component classes dynamically via Hydra (`hydra.utils.get_class(_target_)`), so there are no static cross-package imports.

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

Components live in `vivarium/components/` and follow a three-file pattern:

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

The config structure does not follow [Hydra's recommended pattern](https://hydra.cc/docs/advanced/instantiate_objects/config_files/) where configs mirror `__init__` signatures. Component configs mix constructor args, client-side metadata (`client:` block), and template directives (`_all_values_`, `by_indices`) in the same YAML node, which prevents using `hydra.utils.instantiate()` directly.

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

## Per-Package Documentation

Each package has a `CLAUDE.md` with purpose, structure, API, dependencies, and test coverage:

| Package | CLAUDE.md | Description |
|---------|-----------|--------|
| `vivarium/components/` | [CLAUDE.md](vivarium/components/CLAUDE.md) | Component implementations (entities, physics, eco-evo) |
| `vivarium/environment/` | [CLAUDE.md](vivarium/environment/CLAUDE.md) | Core simulation engine |
| `vivarium/simulator/` | [CLAUDE.md](vivarium/simulator/CLAUDE.md) | State management, gRPC bridge |
| `vivarium/controllers/` | [CLAUDE.md](vivarium/controllers/CLAUDE.md) | User-facing Python API |
| `vivarium/interface/` | [CLAUDE.md](vivarium/interface/CLAUDE.md) | Panel web UI |
| `vivarium/utils/` | [CLAUDE.md](vivarium/utils/CLAUDE.md) | Shared utilities |
| `conf/` | [CLAUDE.md](conf/CLAUDE.md) | Hydra configs |
| `scripts/` | [CLAUDE.md](scripts/CLAUDE.md) | Entry points |
| `tests/` | [CLAUDE.md](tests/CLAUDE.md) | Test suite |
| `notebooks/` | [CLAUDE.md](notebooks/CLAUDE.md) | Jupyter notebooks |

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
