# conf/ — Package Audit

_Audited: 2026-03-10 | Status: Phase 1 Step 1_

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
| `session_6` | Educational | No | **Dead** | Outdated; doesn't set scene_name |

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

## Known Issues

### Should Fix

1. **`session_6.yaml` is dead.** Doesn't set `scene_name`, excluded from all tests, no active references. Delete it.

2. **`braitenberg.yaml` and `particle_lenia.yaml` missing `_self_` in defaults list.** Causes Hydra UserWarning during tests. Add `_self_` to both.

3. **`demo.yaml` has 60+ lines of commented-out code.** Old multi-subtype variants. Clean up.

### Minor

4. **Implicit client inclusion.** `clients/collision.yaml` is included via `base_physics.yaml` but never directly referenced by scene files. Works correctly but not obvious to maintainers.

5. **No config validation.** If a scene defines entity subtypes but doesn't list them in `subtype_labels`, UI controllers may fail silently. Currently caught manually.

## Test Coverage

| Config Group | Test File | What's Tested | Gaps |
|--------------|-----------|---------------|------|
| 10 scene configs | `test_scene_config.py`, `test_environments.py`, `test_edu_sessions.py`, `test_simulator.py` | Config loading, component factories, environment creation, simulation steps | `demo`, `excretion`, `boyds` untested; `miniproject`, `reactive_rl` indirect only |
| Component defaults | Via scene tests | All physics + eco-evo components exercised through scene tests | No isolated component config tests |
| Client configs | Via scene tests | Loaded as part of scene composition | No direct validation tests |

**Gaps**: 3 research scenes (`demo`, `excretion`, `boyds`) have no automated test coverage. Adding them to `test_environments.py` parametrize would be low effort and could catch config errors.

## Refactoring Opportunities

| Priority | Opportunity |
|----------|-------------|
| High | Delete `session_6.yaml` |
| High | Add `_self_` to `braitenberg.yaml` and `particle_lenia.yaml` |
| Medium | Clean commented code in `demo.yaml` |
| Medium | Add test coverage for `demo`, `excretion`, `boyds`, `miniproject`, `reactive_rl` |
| Low | Document implicit client inclusion pattern |
| Low | Consider renaming `clients/` to `controllers/` (high cost, low value — skip for now) |

## Config–Code Structure Mismatch

Hydra's [recommended pattern](https://hydra.cc/docs/advanced/instantiate_objects/config_files/) is: config structure mirrors `__init__` signatures, so `hydra.utils.instantiate(config)` directly constructs objects. Component configs currently mix three concerns in the same YAML node:

1. **Constructor args** (`precedence`, `epsilon`, `alpha`, …) — consumed by `Component.__init__`
2. **Client metadata** (`client:` block with `controller_cls`, `interface_cls`, `controller_kwargs`) — not consumed by the component, filtered out by `Component.get_kwargs()`
3. **Template directives** (`_all_values_`, `_range_`, `_random_`, `by_indices`) — expanded by `EntityComponent.from_config()` before `__init__`

This means `hydra.utils.instantiate()` cannot replace `from_config()` for most classes.

**Example of the problem** — a collision component config node contains both its constructor args and unrelated client metadata:

```yaml
collision:
  _target_: ...CollisionComponent
  name: collision
  precedence: 2          # ← constructor arg
  epsilon: 0.01          # ← constructor arg
  client:                # ← NOT a constructor arg — filtered by get_kwargs()
    controller_cls: ...CollisionController
    controller_kwargs: {collision_alpha: 100.0}
```

**Partial separation already exists:** `clients/*.yaml` files are composed separately via scene defaults lists (e.g., `clients/collision@environment.components.component_list.collision.client`). But they merge *into* the component config node, so the component still sees the `client` key and must filter it.

**Most actionable improvement:** If `client` configs were composed into a parallel structure (e.g., a top-level `clients:` dict) instead of nesting inside component configs, `Component.get_kwargs()` would no longer need to filter, and simple components could use `instantiate()` directly. Entity components would still need `from_config` for template expansion. See also global CLAUDE.md "from_config Pattern" section.

## Structural Questions (resolved in Phase 1 Step 2)

1. **Everything lives under `scene/`.** Resolved: **keep as-is.** Simulator and interface configs are consumed via Hydra composition — they're always loaded as part of a scene. Moving them to `conf/simulator/` would break the composition pattern without benefit. Both `vivarium/simulator/` and `vivarium/interface/` receive their config through the scene's composed config tree, not by loading independent config groups.

2. **Inconsistent naming for "base" configs.** Resolved: **the distinction is intentional.** `default.yaml` files are leaf defaults (used as-is by Hydra's default list mechanism). `base_*.yaml` files are inheritance anchors (meant to be extended by scenes). This is a Hydra convention worth preserving. Document it but don't rename.

3. **Component config organization.** Resolved: **keep current layout.** Current layout (`entities/`, `reset/`, `spawn/` + separate `clients/`) mirrors the code structure in `vivarium/environment/components/`. Regrouping by component name would diverge from code layout for marginal benefit. Not worth the churn.
