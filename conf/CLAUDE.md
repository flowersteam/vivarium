# conf/

## Purpose

Hydra configuration directory. Defines scene compositions, environment parameters, entity/physics/eco-evo components, client (controller+interface) configs, and simulator/interface settings. 48 YAML files total.

## Structure

```
conf/
├── config.yaml                          # Root: sets default scene to "braitenberg"
└── scene/
    ├── base_scene.yaml                  # Template: includes environment, simulator, interface
    ├── braitenberg_defaults.yaml        # Anchor: base_scene + physics + braitenberg entity/client
    ├── session_defaults.yaml            # Anchor: braitenberg_defaults + session environment config
    ├── <scene>.yaml                     # 17 scene files (see table below)
    ├── environment/
    │   ├── default.yaml                 # Base environment template
    │   ├── kwargs/default.yaml          # JAX-MD physics params (box_size, neighbor_radius, etc.)
    │   └── components/
    │       ├── empty.yaml               # Empty component list (fallback)
    │       ├── base_physics.yaml        # Pipeline: reset → collision → friction → step
    │       ├── entities/                # Entity component configs (5 files)
    │       ├── reset/, collision/, friction/, step/  # Physics component defaults
    │       └── energy/, spawn/, consumption/, reproduction/  # Eco-evo defaults
    ├── clients/                         # Controller+interface pair configs (8 files)
    ├── simulator/base_simulator.yaml    # Simulator template
    └── interface/base_interface.yaml    # Panel UI config
```

## Composition Pattern

All scenes follow this inheritance:

```
base_scene
  ↑ braitenberg_defaults (adds physics pipeline + braitenberg entity + clients)
    ↑ session_defaults (adds session-specific environment config)
      ↑ session_1, session_2, ... (add eco-evo components as needed)

  ↑ braitenberg_defaults
    ↑ braitenberg, quickstart, sandbox, demo, boyds, non_transitive, fishing

  base_scene (directly)
    ↑ particle_lenia, excretion, lenia_braitenberg
```

Scenes add eco-evo components via Hydra package directives:
```yaml
defaults:
  - braitenberg_defaults
  - environment/components/spawn/default@environment.components.component_list.spawn
  - clients/spawn@environment.components.component_list.spawn.client
  - _self_
```

## Scenes

| Scene | Category | Tested | Status | Notes |
|-------|----------|--------|--------|-------|
| `braitenberg` | Research | Yes | Solid | Classic baseline; 5 prey + 5 predators |
| `particle_lenia` | Research | Yes | Solid | 600 particles; Lenia dynamics |
| `lenia_braitenberg` | Research | Yes | Solid | Hybrid: 8 Braitenberg + 400 Lenia |
| `non_transitive` | Research | Yes | Solid | Rock-paper-scissors; 3×300 agents |
| `fishing` | Research | Yes | Solid | Food web: boats → fish → plankton |
| `quickstart` | Tutorial | No | Solid | 3 agents + 12 objects |
| `sandbox` | Sandbox | No | Solid | Minimal test environment |
| `demo` | Research | No | Brittle | 184 lines, all features, 60+ lines of commented code |
| `excretion` | Research | No | Brittle | Walls + particles |
| `boyds` | Research | No | Brittle | 300 agents, large scale |
| `session_1` | Educational | Yes | Solid | No eco-evo components |
| `session_2` | Educational | Yes | Solid | Adds objects |
| `session_3` | Educational | Yes (extensive) | Solid | Full eco-evo pipeline |
| `session_4` | Educational | Yes | Solid | Alternate subtype design |
| `miniproject` | Educational | Indirect | Solid | Student template; fixture only |
| `reactive_rl` | Educational | Indirect | Solid | Single agent RL; fixture only |
| `session_6` | Educational | No | Outdated | Doesn't set scene_name |

## Component Precedence Order

| Precedence | Component | Purpose |
|------------|-----------|---------|
| 0 | Reset | Clear forces |
| 2 | Collision | Resolve overlaps |
| 3 | Friction | Dampen velocity |
| 20 | Spawn | Generate new entities |
| 40 | Consumption | Transfer energy via proximity |
| 60 | Energy | Energy decay/burst |
| 80 | Reproduction | Birth/death via energy thresholds |
| 1000 | Step | Euler integration (last) |

## Test Coverage

| Config Group | Test File | What's Tested | Gaps |
|--------------|-----------|---------------|------|
| 10 scene configs | `test_scene_config.py`, `test_environments.py`, `test_edu_sessions.py`, `test_simulator.py` | Config loading, component factories, environment creation, simulation steps | `demo`, `excretion`, `boyds` untested; `miniproject`, `reactive_rl` indirect only |
| Component defaults | Via scene tests | All physics + eco-evo components exercised through scene tests | No isolated component config tests |
| Client configs | Via scene tests | Loaded as part of scene composition | No direct validation tests |

**Gaps**: 3 research scenes (`demo`, `excretion`, `boyds`) have no automated test coverage. Adding them to `test_environments.py` parametrize would be low effort and could catch config errors.

