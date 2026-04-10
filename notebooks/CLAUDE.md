# notebooks/

## Purpose

Jupyter notebooks serving three roles: educational practical sessions (primary, active), tutorials/quickstart (outdated, kept for reference), and server-side headless examples (outdated, kept for reference). The sessions/ directory is the most relevant for now — it provides the main documentation for the Programmatic Pythonic Control API.

## Directory Structure

```
notebooks/
├── README.md                          # Brief overview of notebook categories
├── sandbox.ipynb                      # Development/experimentation sandbox (1.9M)
├── lenia_braitenberg.mp4              # Demo video
├── sessions/                          # Educational practical sessions (ACTIVE)
│   ├── README.md                      # Student setup instructions
│   ├── python_basics.ipynb            # Python refresher (prerequisite)
│   ├── session_1.ipynb                # Basic motor/sensor control
│   ├── session_2.ipynb                # Reactive behaviors (attach/detach)
│   ├── session_3.ipynb                # Selective sensing, consumption, spawning
│   ├── session_4.ipynb                # Internal states, routines, behavior weighting
│   ├── session_5_logging.ipynb        # Data logging and plotting (partial)
│   ├── session_5_logging copy.ipynb   # Duplicate backup
│   ├── session_6_bonus.ipynb          # Eco-evo simulation (OUTDATED — old API)
│   ├── miniproject_template.ipynb     # Student project scaffold (ACTIVE)
│   └── reactive_rl.ipynb             # RL + Braitenberg behaviors (ACTIVE)
├── tutorials/                         # Tutorials (OUTDATED — kept for reference)
│   ├── README.md                      # Minimal overview
│   ├── quickstart_tutorial.ipynb      # Marked outdated; uses old NotebookController
│   ├── troubleshooting.ipynb          # Marked outdated; uses old NotebookController
│   ├── google_colab.ipynb             # Partial; has TODOs, hardcoded branch ref
│   ├── google_colab.md                # Markdown companion
│   └── web_interface_tutorial.md      # Web interface guide (markdown, not notebook)
└── server_side/                       # Headless JAX examples (OUTDATED)
    ├── 1_simple_braitenberg.ipynb     # JAX simulation loop, scaling
    ├── 3_prey_predator_braitenberg.ipynb  # Custom env subclass, JAX vmap/jit
    ├── bokeh_rendering.ipynb          # Old Bokeh rendering (superseded by Panel)
    ├── simulator_tutorial.ipynb       # Simulator API tutorial
    └── outdated_simulator_tutorial.ipynb  # Explicitly labeled outdated
```

## Status by Notebook

### Active & Current (use `VivariumController.start_session()`)

| Notebook | Size | Has Holes | Topic |
|----------|------|-----------|-------|
| `session_1.ipynb` | 30K | Yes (Q1-Q6) | Motor/sensor control basics |
| `session_2.ipynb` | 26K | Yes (Q1-Q3) | Reactive behaviors, attach/detach |
| `session_3.ipynb` | 39K | Yes (Q1-Q5) | Selective sensing, consumption, spawning, prey-predator |
| `session_4.ipynb` | 39K | Yes (Q1-Q4) | Internal states, routines, behavior weighting |
| `miniproject_template.ipynb` | 80K | No | Project scaffold; shows Hydra config customization |
| `reactive_rl.ipynb` | 21K | No | RL + behavior factories, training loop |
| `python_basics.ipynb` | 4.6K | No | enumerate, list comprehension refresher |

### Partial

| Notebook | Status |
|----------|--------|
| `session_5_logging.ipynb` | Partially written. Content exists but not finalized. |
| `session_6_bonus.ipynb` | Outdated. Uses `kill_session()` and old API patterns. |

### Outdated (kept for reference only)

| Notebook | API Pattern |
|----------|-------------|
| `quickstart_tutorial.ipynb` | `NotebookController`, `start_server_and_interface()` |
| `troubleshooting.ipynb` | `NotebookController` |
| `google_colab.ipynb` | Partially current API, has TODOs |
| `1_simple_braitenberg.ipynb` | Direct JAX (`BraitenbergEnv`, `init_state`) |
| `3_prey_predator_braitenberg.ipynb` | Direct JAX (custom env subclass) |
| `bokeh_rendering.ipynb` | Old Bokeh rendering |
| `simulator_tutorial.ipynb` | `Simulator` class direct usage |
| `outdated_simulator_tutorial.ipynb` | Explicitly outdated |

## API Patterns Used

### Current Pattern (sessions 1-4, miniproject, reactive_rl)
```python
from vivarium.controllers import VivariumController
controller = VivariumController.start_session(scene_name="session_1")
agent = controller.agents[0]
agent.left_motor = 1.0
controller.step()
```

### Deprecated Pattern (tutorials, session_6)
```python
from vivarium.controllers.notebook_controller import NotebookController
from vivarium.runtime import stop_server_and_interface
controller = NotebookController()
```

### Server-Side Pattern (server_side/ notebooks), also deprecated
```python
from vivarium.environments.braitenberg.simple.simple_env import BraitenbergEnv, init_state
env = BraitenbergEnv(state)
# Pure JAX simulation loop
```

## Pedagogical Structure

Sessions 1-4 form a coherent progression:
1. **Session 1**: Direct motor/sensor control (for loops, sleep)
2. **Session 2**: Behaviors as functions → `attach_behavior()` / `detach_behavior()`
3. **Session 3**: Selective sensing, multi-agent, consumption/spawning → prey-predator
4. **Session 4**: Internal states (energy), routines, dynamic behavior weight modulation

Each session:
- Uses a dedicated scene config (`scene_name="session_N"`)
- Contains question cells (Q1, Q2...) where students implement code
- Builds on concepts from previous sessions
- Has excellent pedagogical scaffolding with explanations

## What the Notebooks Cover (API-wise)

The active sessions together document the following `VivariumController` API:

| Feature | Session | API |
|---------|---------|-----|
| Start/stop session | 1-4 | `VivariumController.start_session(scene_name=...)` |
| Agent access | 1-4 | `controller.agents[i]`, `controller.objects[i]` |
| Motor control | 1-4 | `agent.left_motor`, `agent.right_motor` |
| Sensor reading | 1-4 | `agent.proximeters()` |
| Selective sensing | 3 | `agent.proximeters(sensed_entities=["obstacle"])` |
| Behaviors | 2-4 | `agent.attach_behavior(fn)`, `detach_behavior(fn)` |
| Behavior weighting | 4 | `agent.attach_behavior(fn, weight=0.5)` |
| Routines | 4 | `agent.attach_routine(fn)`, `detach_routine(fn)` |
| Internal state | 4 | `agent.internal.energy_level` |
| Consumption | 3 | `controller.consumption` settings |
| Spawning | 3 | `controller.spawn` settings |
| Logging | 5 | `agent.logger.add(topic, value)`, `agent.logger.get(topic)` |
| Stepping | 1-4 | `controller.step()` |
| Scene customization | miniproject | Hydra YAML overrides |
| RL integration | reactive_rl | `start_controller_thread=False`, manual stepping |

