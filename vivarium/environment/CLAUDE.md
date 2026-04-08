# vivarium/environment/

## Purpose

Core simulation engine. Defines the state system, `Environment` orchestrator, and JAX math utilities. All simulation logic runs in JAX (pure functions, immutable state, JIT-compatible).

Component implementations live in `vivarium/components/` (see [CLAUDE.md](../components/CLAUDE.md)).

## Structure

```
vivarium/environment/
├── __init__.py              # Exports: Environment, NeighborManager, MaskFunction
├── state.py                 # JAX-MD state classes, create_state_cls(), field_accessors
├── environment.py           # Environment orchestrator: init, step, component composition
├── utils.py                 # JAX math helpers: distance, proximity_map, type_mask, etc.
└── render.py                # Matplotlib visualization (testing/notebooks only)
```

## Architecture

### State System

`create_state_cls()` dynamically builds the state class:
1. Starts from `BaseState` (time + entity_state)
2. Each component calls `update_state_cls()` to add fields
3. Result is a JAX-MD dataclass (immutable, JIT-compatible)

`BaseEntityState` (flat array for all entities): position, momentum, mass, force, orientation, entity_type, entity_type_idx, entity_subtype, exists, diameter, friction.

Component-specific state is added as named fields (e.g., `state.agents`, `state.spawn_state`).

## Public API

### From `__init__.py`
- `Environment` — main orchestrator
- `NeighborManager` — JAX-MD neighbor list wrapper
- `MaskFunction` — mask factory (currently only supports `'exists'` label)

### Key classes used by other packages
- `Environment`, `NeighborManager`, `MaskFunction` — used by `simulator/`, tests
- State classes (`BaseState`, `BaseEntityState`, `create_state_cls`) — used by components, tests

## Test Coverage

| Area | Test File | What's Tested | Gaps |
|------|-----------|---------------|------|
| Environment creation | `test_environment.py` | `from_config()`, stepping, state shape | Tested for 5 scenes (braitenberg, particle_lenia, lenia_braitenberg, non_transitive, fishing) |
| State system | `test_state.py` | State creation, field accessors | Solid |
| Edu sessions | `test_edu_sessions.py` | Session configs create valid environments | Solid |

**Gaps:**
- `render.py` — untested (and has bugs)
