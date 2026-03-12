# notebooks/ — Package Audit

_Audited: 2026-03-11 | Status: Phase 1 Step 1_

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
│   ├── session_5_logging.ipynb        # Data logging and plotting (INCOMPLETE)
│   ├── session_5_logging copy.ipynb   # Duplicate/backup (DELETE)
│   ├── session_6_bonus.ipynb          # Eco-evo simulation (OUTDATED — old API)
│   ├── miniproject_template.ipynb     # Student project scaffold (ACTIVE)
│   └── reactive_rl.ipynb             # RL + Braitenberg behaviors (ACTIVE)
├── tutorials/                         # Tutorials (OUTDATED — kept for reference)
│   ├── README.md                      # Minimal overview
│   ├── quickstart_tutorial.ipynb      # Marked outdated; uses old NotebookController
│   ├── troubleshooting.ipynb          # Marked outdated; uses old NotebookController
│   ├── google_colab.ipynb             # Incomplete; has TODOs, hardcoded branch ref
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

### Needs Work

| Notebook | Issue |
|----------|-------|
| `session_5_logging.ipynb` | Marked "still has to be updated". Content exists but incomplete. |
| `session_6_bonus.ipynb` | Marked outdated. Uses `kill_session()`, old APIs. Needs full rewrite to current API. `miniproject_template.ipynb` provides a good basis for the rewrite. |

### Outdated (kept for reference only)

| Notebook | API Pattern |
|----------|-------------|
| `quickstart_tutorial.ipynb` | `NotebookController`, `start_server_and_interface()` |
| `troubleshooting.ipynb` | `NotebookController` |
| `google_colab.ipynb` | Partially current API, but incomplete with TODOs |
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
from vivarium.utils.handle_server_interface import start_server_and_interface
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

## Known Issues

### Should Fix

1. **`session_5_logging copy.ipynb` is a duplicate.** 284K backup file that should be deleted.

2. **Session 5 incomplete.** Marked "still has to be updated". Needs review and completion.

3. **Session 6 uses deprecated APIs.** References `kill_session()`, old controller patterns. Needs full rewrite using `miniproject_template.ipynb` as basis for current API patterns.

4. **`google_colab.ipynb` has hardcoded branch reference** (`clement/revive-notebook-controller`). Should reference `main` or be parameterized.

### Medium Severity

5. **No notebook testing mechanism.** No way to verify notebooks still work after code changes. The "holes" pattern (empty cells for students) makes standard notebook testing tools (nbval, pytest-notebook) inadequate.

6. **`sandbox.ipynb` is 1.9M.** Large file with embedded outputs. Should be cleared or gitignored.

7. **Tutorials folder has no clear purpose post-rewrite.** All three notebooks are outdated. Need to decide: update, replace, or archive.

### Low Severity

8. **Server-side notebooks reference old import paths.** `vivarium.environments.braitenberg.simple.simple_env` no longer exists in current codebase.

9. **No notebook execution order enforcement.** Students could run session 3 before session 1 without error, but concepts won't make sense.

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

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Delete `session_5_logging copy.ipynb` |
| High | Complete session 5 |
| High | Rewrite session 6 with current API (using miniproject_template as reference) |
| High | Decide on tutorials/ folder: update `quickstart_tutorial.ipynb` as API entry point or write new |
| Medium | Clear sandbox.ipynb outputs or add to .gitignore |
| Medium | Fix google_colab.ipynb branch reference and TODOs |
| Medium | Design notebook testing strategy (see Structural Questions) |
| Low | Archive or remove server_side/ notebooks |
| Low | Add session dependency notes to README |

## Structural Questions (updated in Phase 1 Step 2)

1. **Notebook documentation structure.** Deferred to Phase 3. The active sessions cover the Programmatic Pythonic Control API well. A rewritten `quickstart_tutorial.ipynb` could serve as a standalone getting-started guide (vs. the progressive sessions). Decision depends on Phase 3 documentation scope.

2. **Notebook testing strategy.** Deferred to Phase 3. See tests/CLAUDE.md for options.

3. **Server-side notebook future.** These are the only documentation for the headless JAX workflow (audience 1: researchers). Import paths are all obsolete. Rewriting at least one (e.g. `1_simple_braitenberg.ipynb`) for the current API is needed for the researcher audience. Decision depends on Phase 3 scope.

4. **Session 6 scope.** `miniproject_template.ipynb` already covers most of what session 6 does (eco-evo dynamics, custom configs). Recommend: rewrite session 6 as a lighter complement to the miniproject (focused on eco-evo concepts) rather than a standalone session. Or merge into miniproject. Decision for Phase 3.
