import jax
import jax.numpy as jnp

from jax_md import partition

from vivarium.environment.utils import type_mask, neighbors_entity_mask
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


def test_spawn(environment_and_state, spawn):
    env, state = environment_and_state(spawn)
    # SpawnState fields are now 2D [n_configs, ...]; index into config 0
    assert state.spawn_state.start[0]

    for _ in range(4):
        assert state.entity_state.exists.sum() == state.entity_state.exists.shape[0]
        state  = state.set(
            entity_state=state.entity_state.set(
                exists=state.entity_state.exists.at[0].set(0)
            )
        )
        assert state.entity_state.exists.sum() == state.entity_state.exists.shape[0] - 1
        prev_pos_0 = state.entity_state.position[0]
        prev_orientation_0 = state.entity_state.orientation[0]
        state = env.step(state)
        assert state.entity_state.exists.sum() == state.entity_state.exists.shape[0]
        assert state.entity_state.position[0, 0] >= state.spawn_state.position_range[0, 0]
        assert state.entity_state.position[0, 0] <= state.spawn_state.position_range[0, 1]
        assert state.entity_state.position[0, 1] >= state.spawn_state.position_range[0, 2]
        assert state.entity_state.position[0, 1] <= state.spawn_state.position_range[0, 3]
        assert state.entity_state.orientation[0] >= state.spawn_state.orientation_range[0, 0]
        assert state.entity_state.orientation[0] <= state.spawn_state.orientation_range[0, 1]
        assert not jnp.equal(prev_pos_0, state.entity_state.position[0]).all()
        assert not jnp.equal(prev_orientation_0, state.entity_state.orientation[0]).all()
    

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
    
    mask = neighbors_entity_mask(
        neighbors_idx=env.neighbor_manager.neighbors.idx,
        source_mask=jnp.full(state.entity_state.exists.shape, False, dtype=bool).at[consumer_idx].set(True),
        target_mask=jnp.full(state.entity_state.exists.shape, False, dtype=bool).at[consumee_idx].set(True),
        neighbor_mask=partition.neighbor_list_mask(env.neighbor_manager.neighbors)
    )
    
    assert mask.sum() == 1
    
    consumption_idx = [i.item() for i in jnp.nonzero(mask)]
    assert state.consumption_state.consumption_matrix[*consumption_idx] == 0.
    

    state = env.step(state, scan=False)

    consumption_idx = [i.item() for i in jnp.nonzero(mask)]
    assert state.consumption_state.consumption_matrix[*consumption_idx] > 0.

    consuming = state.consumption_state.consumption_matrix.sum(axis=1)
    
    neigh_flat = env.neighbor_manager.neighbors.idx.ravel()
    consumption_matrix_flat = state.consumption_state.consumption_matrix.ravel()
    
    consumed = jax.ops.segment_sum(consumption_matrix_flat, neigh_flat) #, n_entities)      
    
    assert consumed.sum() == consuming.sum()
    
    


def test_energy(environment_and_state, energy):
    env, state = environment_and_state(energy)
    idx = 0
    state = state.set(
        consumption_state=state.consumption_state.set(
            consumption_matrix=jnp.full(state.consumption_state.consumption_matrix.shape, 0.).at[idx, 2].set(1.),
        )
    )
    energy_step_fn = env.get_factory_by_name('energy').get_step_function(state, env.neighbor_manager, None)
    state = energy_step_fn(state, env.neighbor_manager.neighbors, None)
    new_energy = state.entity_state.energy[idx]
    assert new_energy == 1


def test_death(environment_and_state, reproduction):
    env, state = environment_and_state(reproduction)
    idx = 0
    etype_idx = state.entity_state.entity_type_idx[idx]
    state = env.step(state)
    state = state.set(
        entity_state=state.entity_state.set(
            energy=state.entity_state.energy.at[etype_idx].set(1.),
        )
    )
    state = env.step(state)
    assert state.entity_state.exists[idx] == 1
    state = state.set(
        entity_state=state.entity_state.set(
            energy=state.entity_state.energy.at[etype_idx].set(0.),
        )
    )
    state = env.step(state)
    assert state.entity_state.exists[idx] == 0

def test_reproduction(environment_and_state, reproduction):
    env, state = environment_and_state(reproduction, debug_mode=True)
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
        entity_state=state.entity_state.set(
            energy=state.entity_state.energy.at[idx].set(1.),
        ),
        agents=state.agents.set(
            reproduction=state.agents.reproduction.set(
                recover_time=state.agents.reproduction.recover_time.at[etype_idx].set(1e5),
            )
        )
    )
    state = state = env.step(state)
    assert state.entity_state.exists.sum() == n_exists + 1


def test_braitenberg(environment_and_state, braitenberg):
    env, state = environment_and_state(braitenberg)
    state = env.step(state)
    pass
