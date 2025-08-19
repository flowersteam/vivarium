import hydra
import jax.numpy as jnp

from vivarium.environments.utils import type_mask
from vivarium.utils.scene_configs import component_factories_from_config

def test_instantiate(scene_config):
    scene_config = scene_config('braitenberg')
    component_factories = component_factories_from_config(scene_config.environment.components)
    pass


def test_type_mask(environment_and_state, braitenberg):
    _, state = environment_and_state(braitenberg)
    entity_state = state.entity_state
    exists = jnp.zeros_like(entity_state.exists)
    idx = 3
    exists = exists.at[idx].set(1)
    entity_state = entity_state.set(exists=exists)
    exists = exists.at[idx + 2].set(1)
    mask = type_mask(entity_state,
                     exists=0,
                     entity_type=entity_state.entity_type[idx],
                     subtype=entity_state.entity_subtype[idx])
    assert jnp.equal(mask, 
                     jnp.logical_and(
                         entity_state.exists == 0,
                         jnp.logical_and(
                            entity_state.entity_type == entity_state.entity_type[idx],
                            entity_state.entity_subtype == entity_state.entity_subtype[idx]
                            )
                        )
                     ).all()
    
    mask = type_mask(entity_state,
                     exists=1,
                     entity_type=entity_state.entity_type[idx],
                     subtype=entity_state.entity_subtype[idx])
    assert jnp.equal(mask, 
                     jnp.logical_and(
                         entity_state.exists == 1,
                         jnp.logical_and(
                            entity_state.entity_type == entity_state.entity_type[idx],
                            entity_state.entity_subtype == entity_state.entity_subtype[idx]
                            )
                        )
                     ).all()

    mask = type_mask(entity_state,
                     exists=0,
                     entity_type=entity_state.entity_type[idx]
                     )
    assert jnp.equal(mask, 
                     jnp.logical_and(
                         entity_state.exists == 0,
                         entity_state.entity_type == entity_state.entity_type[idx]
                         )
                     ).all()
    
    mask = type_mask(entity_state,
                     exists=0,
                     subtype=entity_state.entity_subtype[idx])
    assert jnp.equal(mask, 
                     jnp.logical_and(
                         entity_state.exists == 0,
                         entity_state.entity_subtype == entity_state.entity_subtype[idx]
                         )
                     ).all()


def test_proximity_map(environment_and_state, proximity_map):
    env, state = environment_and_state(proximity_map)
    state = env.step(state)


def test_consumption(environment_and_state, consumption):

    env, state = environment_and_state(consumption)

    consumer_idx = 0
    consumee_idx = 3

    pos_consumee = state.entity_state.position_center[consumer_idx] + jnp.ones(2)
    state = state.set(
        entity_state=state.entity_state.set(
            position=state.entity_state.position.at[consumee_idx].set(pos_consumee)
        )
    )
    
    assert not state.entity_state.consuming[consumer_idx]
    assert not state.entity_state.consumed[consumee_idx]
    assert state.entity_state.exists[consumee_idx]

    state = env.step(state, scan=False)

    assert state.entity_state.consuming[consumer_idx]
    assert state.entity_state.consumed[consumee_idx]
    assert not state.entity_state.exists[consumee_idx]


def test_energy(environment_and_state, energy):
    env, state = environment_and_state(energy)
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    state = state.set(
        entity_state=state.entity_state.set(
            consuming=state.entity_state.consuming.at[idx].set(True),
        )
    )
    state = env.step(state, scan=False)
    new_energy = state.agents.energy[etype_idx]
    assert new_energy == 1


def test_death(environment_and_state, reproduction):
    env, state = environment_and_state(reproduction)
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    state = env.step(state)
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(1.),
        )
    )
    state = env.step(state)
    assert state.entity_state.exists[idx] == 1
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(0.),
        )
    )
    state = env.step(state)
    assert state.entity_state.exists[idx] == 0

def test_reproduction(environment_and_state, reproduction):
    env, state = environment_and_state(reproduction)
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    state = state.set(
        entity_state=state.entity_state.set(
            exists=state.entity_state.exists.at[idx+1].set(0),
        )
    )
    n_exists = state.entity_state.exists.sum()
    state = env.step(state, scan=False)
    assert state.entity_state.exists.sum() == n_exists
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(1.),
            recover_time=state.agents.recover_time.at[etype_idx].set(1e5),
        )
    )
    state = state = env.step(state)
    assert state.entity_state.exists.sum() == n_exists + 1


def test_braitenberg(environment_and_state, braitenberg):
    env, state = environment_and_state(braitenberg)
    state = env.step(state)
    pass
