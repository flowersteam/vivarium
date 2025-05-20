import pytest
import jax.numpy as jnp
from vivarium.environments.physics_engine import ProximityMap
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.environments.dynamics.existence import type_mask

@pytest.fixture
def scene_config():
    scene_name = 'braitenberg'
    return SceneConfiguration(scene_name)
@pytest.fixture
def state(scene_config):
    return scene_config.create_state()
@pytest.fixture
def env(scene_config):
    return scene_config.create_environment()

def test_type_mask(state):
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

def test_proximity_map(scene_config):
    map = ProximityMap('test', 0)
    map.update_scene_configuration(scene_config)
    env = scene_config.create_environment()
    env.dynamics_functions = [map.get_state_function(env)]
    state = env.dynamics_functions[0](env.state, env.neighbor_manager.neighbors, env.key)
    state

def test_consumption(env):
    # env = SceneConfiguration('braitenberg').create_environment()
    state = env.state
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    # state = state.set(
    #     entity_state=state.entity_state.set(
    #         consuming=state.entity_state.consuming.at[idx].set(True),
    #     )
    # )
    pos_resource = state.entity_state.position_center[idx] + jnp.ones(2)
    state = state.set(
        entity_state=state.entity_state.set(
            position=state.entity_state.position.at[-1].set(pos_resource)
        )
    )
    proximity_map_fn = env.get_dynamics_function_by_name('proximity_map')
    state = proximity_map_fn(state, env.neighbor_manager.neighbors, env.key)
    consumption_fn = env.get_dynamics_function_by_name('preys_consume_resources')
    state = consumption_fn(state, env.neighbor_manager.neighbors, env.key)
    # energy_fn = env.dynamics_functions[-5]
    # state = energy_fn(state, env.neighbor_manager.neighbors, env.key)
    # reproduction_fn = env.dynamics_functions[-4]
    # state = reproduction_fn(state, env.neighbor_manager.neighbors, env.key)
    state


def test_energy_routine(env):
    # env = SceneConfiguration('braitenberg').create_environment()
    state = env.state
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    cur_energy = state.agents.energy[etype_idx]
    state = state.set(
        entity_state=state.entity_state.set(
            consuming=state.entity_state.consuming.at[idx].set(True),
        )
    )
    energy_fn = env.get_dynamics_function_by_name('energy_preys')
    state = energy_fn(state, env.neighbor_manager.neighbors, env.key)
    new_energy = state.agents.energy[etype_idx]
    assert new_energy == 1


def test_death(env):
    # env = SceneConfiguration('braitenberg').create_environment()
    state = env.state
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    reproduction_fn = env.get_dynamics_function_by_name('reproduction')
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(1.),
        )
    )
    state = reproduction_fn(state, env.neighbor_manager.neighbors, env.key)
    assert state.entity_state.exists[idx] == 1
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(0.),
        )
    )
    state = reproduction_fn(state, env.neighbor_manager.neighbors, env.key)
    assert state.entity_state.exists[idx] == 0

def test_reproduction(env):
    # env = SceneConfiguration('braitenberg').create_environment()
    state = env.state
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    reproduction_fn = env.get_dynamics_function_by_name('reproduction')
    state = state.set(
        entity_state=state.entity_state.set(
            exists=state.entity_state.exists.at[idx+1].set(0),
        )
    )
    n_exists = state.entity_state.exists.sum()
    state = reproduction_fn(state, env.neighbor_manager.neighbors, env.key)
    assert state.entity_state.exists.sum() == n_exists
    state = state.set(
        agents=state.agents.set(
            energy=state.agents.energy.at[etype_idx].set(1.),
            recover_time=state.agents.recover_time.at[etype_idx].set(1e5),
        )
    )
    state = reproduction_fn(state, env.neighbor_manager.neighbors, env.key)
    assert state.entity_state.exists.sum() == n_exists + 1