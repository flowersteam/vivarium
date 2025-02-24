import jax.numpy as jnp

from vivarium.controllers.dataclass_wrapper import *

from vivarium.environments.braitenberg.selective_sensing import (
    init_state as init_rigid_body_state, 
    EntityType)

from vivarium.utils.scene_configs import load_scene_config
import pytest

from vivarium.simulator.simulator import env_to_sim_state

from vivarium.environments.braitenberg import selective_sensing
from vivarium.environments.utils import rigid_body_to_point_particle

init_state_point_particle, _ = rigid_body_to_point_particle(selective_sensing)




def get_rigid_body_state():
    config = load_scene_config('prey_predator')
    return init_rigid_body_state(**config)

def get_point_particle_state():
    config = load_scene_config('prey_predator')
    return init_state_point_particle(**config)

def generate_changes(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value):
    return {
        'agent_state': {'wheel_diameter': [{'__idx': wheel_diameter_idx, '__value': wheel_diameter_value}]},
        'entity_state': {
            'exists': [{'__idx': exists_idx, '__value': exists_value}],
            'friction': [{'__idx': friction_idx, '__value': friction_value},
                         {'__idx': slice(friction_idx, friction_idx + 2), '__value': friction_value + 1}]
        }
    }

def generate_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value):
    return {
        'agent_state': {'wheel_diameter': lambda state: state.agent_state.wheel_diameter.at[wheel_diameter_idx].set(wheel_diameter_value)},
        'entity_state': {
            'exists': lambda state: state.entity_state.exists.at[exists_idx].set(exists_value),
            'friction': lambda state: state.entity_state.friction.at[friction_idx].set(friction_value).at[friction_idx:friction_idx+2].set(friction_value + 1)
        }
    }

def generate_changes_and_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value):
    changes = generate_changes(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value)
    expected = generate_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value)
    return changes, expected

@pytest.mark.parametrize("changes_and_expected, state_fn", [
    (generate_changes_and_expected(0, 0, 7, 0, 6, 0), get_rigid_body_state),
    (generate_changes_and_expected(1, 1, 8, 1, 7, 1), get_point_particle_state),
])
def test_change_recorder(changes_and_expected, state_fn):
    state = state_fn()

    changes, expected = changes_and_expected

    c = ChangeRecorder()

    for entity, attrs in changes.items():
        for attr, changes_list in attrs.items():
            for change in changes_list:
                if change['__idx'] is None:
                    setattr(getattr(c, entity), attr, change['__value'])
                else:
                    getattr(getattr(c, entity), attr)[change['__idx']] = change['__value']

    fetched_changes = c.fetch_changes()

    assert (not c._children)

    for entity, attrs in changes.items():
        for attr, changes_list in attrs.items():
            assert len(fetched_changes[entity][attr]) == len(changes_list)
            for i, change in enumerate(changes_list):
                assert fetched_changes[entity][attr][i]['__idx'] == change['__idx']
                assert fetched_changes[entity][attr][i]['__value'] == change['__value']

    state = update_state(state, fetched_changes)

    for entity, attrs in expected.items():
        for attr, expected_fn in attrs.items():
            assert (jnp.equal(getattr(getattr(state, entity), attr), expected_fn(state))).all()


def test_change_recorder_simstate():
    env_state = get_rigid_body_state()
    state = env_to_sim_state(env_state, num_steps_lax=2, freq=60, use_fori_loop=False, jit_step=False)

    assert (state.simulator_state.freq[0] == 60)

    change_recorder = ChangeRecorder()
    change_recorder.simulator_state.freq[0] = -1
    changes = change_recorder.fetch_changes()

    state = update_state(state, changes)

    assert (state.simulator_state.freq[0] == -1)


def test_dataclass_wrapper():
    state = get_rigid_body_state()
    idx = 3
    val = [0.2, 0.3]
    cur_val = state.agent_state.motor[idx]
    assert (not jnp.equal(jnp.array(cur_val), jnp.array(val)).all())

    state = DataclassWrapper().agent_state.motor[idx].set(val).apply(state)

    assert (jnp.equal(jnp.array(state.agent_state.motor[idx]), jnp.array(val)).all())


@pytest.mark.parametrize("idx, position_center, position_orientation, color, init_state_fn, entity_type", [
    (2, [7, 8], 2.0, [1.0, 0.0, 1.0], get_rigid_body_state, EntityType.AGENT),
    (3, [5, 6], 1.5, [0.5, 0.5, 0.5], get_point_particle_state, EntityType.OBJECT),
    (4, [9, 10], 3.0, [0.0, 1.0, 0.0], get_rigid_body_state, EntityType.AGENT),
])
def test_entity_wrapper(idx, position_center, position_orientation, color, init_state_fn, entity_type):
    state = init_state_fn()

    entity = EntityWrapper(state, idx, entity_type)

    entity.position_center = position_center
    entity.position_orientation = position_orientation
    entity.color = color

    previous_state = state

    state = entity.apply_to_state(state)

    previous_entity_state = previous_state.agent_state if entity_type == EntityType.AGENT else previous_state.object_state
    entity_state = state.agent_state if entity_type == EntityType.AGENT else state.object_state

    assert (not (jnp.equal(previous_state.entity_state.position_center[idx], jnp.array(position_center))).all())
    assert (jnp.equal(state.entity_state.position_center[idx], jnp.array(position_center))).all()
    assert (jnp.equal(state.entity_state.position_center, previous_state.entity_state.position_center.at[idx].set(position_center))).all()
    assert (jnp.equal(state.entity_state.position_orientation, previous_state.entity_state.position_orientation.at[idx].set(position_orientation))).all()
    assert (jnp.equal(entity_state.color, previous_entity_state.color.at[idx].set(color))).all()


@pytest.mark.parametrize("idx, position_center, color, entity_type, state_fn", [
    (4, [7, 8], [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]], EntityType.AGENT, get_rigid_body_state),
    (5, [9, 10], [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0]], EntityType.OBJECT, get_point_particle_state),
])
def test_entity_list(idx, position_center, color, entity_type, state_fn):
    state = state_fn()

    objects = EntityList(state, entity_type)

    obj = objects[idx]
    obj.position_center = position_center

    state = objects.apply_to_state(state)

    entity_state = state.agent_state if entity_type == EntityType.AGENT else state.object_state
    expected_position = state.entity_state.position_center.at[entity_state.ent_idx[idx]].set(position_center)

    for i, o in enumerate(objects):
        if i > len(color) - 1:
            break
        o.color = color[i]
    
    state = objects.apply_to_state(state)
    entity_state = state.agent_state if entity_type == EntityType.AGENT else state.object_state

    assert (jnp.equal(state.entity_state.position_center, expected_position)).all()
    for i, c in enumerate(color):
        assert (jnp.equal(entity_state.color[i], jnp.array(c))).all()
