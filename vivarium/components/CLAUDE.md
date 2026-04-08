# vivarium/components/

## Purpose

Component implementations for the simulation. Each component encapsulates a simulation feature (entity type, physics rule, or eco-evolutionary mechanic) with up to three files: JAX step function (server-side), controller (client-side API), and interface (Panel UI).

## Structure

```
vivarium/components/
├── __init__.py              # Re-exports subpackage modules
├── component.py             # Base Component class (all components inherit from this)
├── interface.py             # Abstract Renderer & Interface base classes (Panel UI)
├── utils.py                 # Force summation, SPACE_NDIMS, count_masked_values
├── entities/
│   ├── component.py         # EntityComponent base (shared init for all entity types)
│   ├── controller.py        # EntityWrapper, EntityController, EntityListController
│   ├── interface.py         # EntityRenderer, EntityInterface (Panel drag-drop, selection)
│   ├── braitenberg/         # Agent type: sensorimotor, behaviors, proximeters
│   ├── objects/             # Passive objects (no behavior, physics only)
│   ├── particle_lenia/      # Particle Lenia creatures (energy-based dynamics)
│   └── walls/               # Static wall obstacles with segment collision
├── physics/
│   ├── reset/               # ResetForceComponent (precedence 0)
│   ├── collision/           # CollisionComponent (soft sphere, precedence 2)
│   ├── friction/            # FrictionComponent (precedence 3)
│   └── step/                # StepComponent (Verlet integration, precedence 1000)
├── eco_evo/
│   ├── utils.py             # spawn_entity, non_existing, sample_true_index
│   ├── spawn/               # SpawnComponent (multi-spawn, precedence 20)
│   ├── consumption/         # ConsumptionComponent (feeding matrix, precedence 40)
│   ├── energy/              # EnergyComponent (decay/burst/clamp, precedence 60)
│   └── reproduction/        # ReproductionComponent (birth/death thresholds, precedence 80)
└── proximity_map/           # ProximityMapComponent (distance & angle maps)
```

## Architecture

### Component Pattern

Each component follows a three-file pattern:

```
component_name/
  ├── component.py      # Component subclass: JAX step function (server-side, pure functional)
  ├── controller.py     # Client-side Python API (getattr/setattr → state and controller parameters paths)
  └── interface.py      # Panel UI (Bokeh renderer + param controls)
```

Not all components have all three files — simpler ones (reset, friction, energy, reproduction) only have `component.py`.

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

### Entity Types

| Type | Component | State Fields | Has Controller | Has Interface |
|------|-----------|-------------|----------------|---------------|
| Braitenberg agents | `braitenberg/component.py` | prox, motor, behavior_params, wheel_diameter, ... | Yes (AgentController, BehaviorController) | Yes |
| Objects | `objects/component.py` | entity_idx only | Via EntityController | Via EntityInterface |
| Particle Lenia | `particle_lenia/component.py` | mu, sigma, w (kernel params) | No | Yes (minimal) |
| Walls | `walls/component.py` | col_start, col_end, wall_type | Yes (minimal) | Yes (minimal) |

## Dependencies

- **Imports from:** `vivarium.environment` (state classes, Environment), `vivarium.controllers` (Controller, AttributeMapping, handlers), `vivarium.utils` (dataclass_wrapper)
- **Imported by:** `vivarium.environment` (dynamically via Hydra `_target_`), tests, scripts

## Key Classes

- `Component` (`component.py`) — base class for all components
- `EntityComponent` (`entities/component.py`) — base class for entity-type components
- `Renderer`, `Interface` (`interface.py`) — abstract base classes for Panel UI
- `EntityController`, `EntityListController` (`entities/controller.py`) — base controllers for entity types

## Test Coverage

| Area | Test File | What's Tested | Gaps |
|------|-----------|---------------|------|
| Components | `test_components.py` | Component instantiation, `type_mask`, `neighbors_entity_mask` | No isolated tests for individual component step functions |
| Multi-spawn | `test_multi_spawn.py` | SpawnComponent with multiple configs | Solid (dedicated tests) |
| Entity controllers | `test_dataclass_wrapper.py` | EntityWrapper, EntityList | Solid |
| Braitenberg behaviors | `test_vivarium_controller.py` | Behavior params | Indirect |

**Gaps:**
- Reproduction logic — no dedicated tests
- Energy distribution algorithm — untested in isolation
- Consumption matrix computation — untested in isolation
- Wall geometry (`segment_point_distance`) — untested
- ProximityMapComponent — exercised via conftest fixture but no dedicated tests
- Braitenberg sensorimotor in isolation — untested
