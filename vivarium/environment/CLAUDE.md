# vivarium/environment/

## Purpose

Core simulation environment. Defines the state system, component architecture, entity types, physics pipeline, and eco-evolutionary dynamics. All simulation logic runs in JAX (pure functions, immutable state, JIT-compatible). ~56 files, ~150KB total.

## Structure

```
vivarium/environment/
├── __init__.py              # Exports: Environment, NeighborManager, MaskFunction
├── state.py                 # JAX-MD state classes, create_state_cls(), field_accessors
├── environment.py           # Environment orchestrator: init, step, component composition
├── utils.py                 # JAX math helpers: distance, proximity_map, type_mask, etc.
├── render.py                # Matplotlib visualization (testing/notebooks only)
└── components/
    ├── component.py         # Base Component class (all components inherit from this)
    ├── utils.py             # Force summation, SPACE_NDIMS, count_masked_values
    ├── interface.py         # Abstract Renderer & Interface base classes (Panel UI)
    ├── entities/
    │   ├── component.py     # EntityComponent base (shared init for all entity types)
    │   ├── controller.py    # EntityWrapper, EntityController, EntityListController
    │   ├── interface.py     # EntityRenderer, EntityInterface (Panel drag-drop, selection)
    │   ├── braitenberg/     # Agent type: sensorimotor, behaviors, proximeters
    │   ├── objects/         # Passive objects (no behavior, physics only)
    │   ├── particle_lenia/  # Particle Lenia creatures (energy-based dynamics)
    │   └── walls/           # Static wall obstacles with segment collision
    ├── physics/
    │   ├── reset/           # ResetForceComponent (precedence 0)
    │   ├── collision/       # CollisionComponent (soft sphere, precedence 2)
    │   ├── friction/        # FrictionComponent (precedence 3)
    │   └── step/            # StepComponent (Verlet integration, precedence 1000)
    ├── eco_evo/
    │   ├── component.py     # Empty placeholder
    │   ├── utils.py         # spawn_entity, non_existing, sample_true_index
    │   ├── spawn/           # SpawnComponent (multi-spawn, precedence 20)
    │   ├── consumption/     # ConsumptionComponent (feeding matrix, precedence 40)
    │   ├── energy/          # EnergyComponent (decay/burst/clamp, precedence 60)
    │   └── reproduction/    # ReproductionComponent (birth/death thresholds, precedence 80)
    └── proximity_map/       # ProximityMapComponent (distance & angle maps)
```

## Architecture

### State System

`create_state_cls()` dynamically builds the state class:
1. Starts from `BaseState` (time + entity_state)
2. Each component calls `update_state_cls()` to add fields
3. Result is a JAX-MD dataclass (immutable, JIT-compatible)

`BaseEntityState` (flat array for all entities): position, momentum, mass, force, orientation, entity_type, entity_type_idx, entity_subtype, exists, diameter, friction.

Component-specific state is added as named fields (e.g., `state.agents`, `state.spawn_state`).

### Component Lifecycle

**Init (once):**
1. `Component.from_config(config)` — instantiate from Hydra config
2. `update_state_cls(state_cls)` — add fields to state class
3. `init_base_entity(entity_state)` — for entities: populate entity_state arrays
4. `init_state_fn(state, neighbor_manager, key)` — initialize component-specific fields
5. `get_step_function(state, neighbor_manager, key)` — return pure JAX step function

**Step (each tick):**
- `Environment.step(state)` runs step functions in precedence order (lower first)
- Each function: `fn(state, neighbors, key) → state` (pure, no side effects)
- JIT'd if `to_jit=True`

### Component Pattern

Each component follows:
```
component_name/
  ├── component.py      # JAX simulation logic
  ├── controller.py     # Client-side API (getattr/setattr mapping to state paths)
  └── interface.py      # Panel UI (Renderer for visualization, Interface for controls)
```

Not all components have all three files — simpler ones (reset, friction, energy, reproduction) only have `component.py`.

### Entity Types

| Type | Component | State Fields | Has Controller | Has Interface |
|------|-----------|-------------|----------------|---------------|
| Braitenberg agents | `braitenberg/component.py` | prox, motor, behavior_params, wheel_diameter, ... | Yes (AgentController, BehaviorController) | Yes |
| Objects | `objects/component.py` | entity_idx only | Via EntityController | Via EntityInterface |
| Particle Lenia | `particle_lenia/component.py` | mu, sigma, w (kernel params) | No | Yes (minimal) |
| Walls | `walls/component.py` | col_start, col_end, wall_type | Yes (minimal) | Yes (minimal) |

## Public API

### From `__init__.py`
- `Environment` — main orchestrator
- `NeighborManager` — JAX-MD neighbor list wrapper
- `MaskFunction` — mask factory (currently only supports `'exists'` label)

### Key classes used by other packages
- `EntityComponent`, `EntityController`, `EntityListController` — used by `controllers/`
- `Renderer`, `Interface` — used by `interface/`
- `BraitenbergComponent`, `ObjectComponent` — used by tests and configs
- `SpawnComponent`, `ConsumptionComponent` — used by tests

## Test Coverage

| Area | Test File | What's Tested | Gaps |
|------|-----------|---------------|------|
| Environment creation | `test_environments.py` | `from_config()`, stepping, state shape | Tested for 5 scenes (braitenberg, particle_lenia, lenia_braitenberg, non_transitive, fishing) |
| State system | `test_state.py` | State creation, field accessors | Solid |
| Components | `test_components.py` | Component instantiation, `type_mask`, `neighbors_entity_mask` | No isolated tests for individual component step functions |
| Multi-spawn | `test_multi_spawn.py` | SpawnComponent with multiple configs | Solid (dedicated tests) |
| Entity controllers | `test_dataclass_api.py` | EntityWrapper, EntityList | Solid |
| Braitenberg behaviors | `test_vivarium_controller.py` | Behavior params | Indirect |
| Edu sessions | `test_edu_sessions.py` | Session configs create valid environments | Solid |

**Gaps:**
- Reproduction logic — complex state-dependent birth/death, no dedicated tests (`test_reproduction` has a pre-existing failure per MEMORY.md)
- Energy distribution algorithm — untested in isolation
- Consumption matrix computation — untested in isolation
- Wall geometry (`segment_point_distance`) — untested
- `render.py` — untested (and has bugs)
- ProximityMapComponent — exercised via conftest fixture but no dedicated tests
- Braitenberg sensorimotor in isolation — untested

