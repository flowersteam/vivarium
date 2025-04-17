import pytest
import jax.numpy as jnp

from vivarium.controllers.dataclass_wrapper import *
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.environments.state import to_rigid_body_state
from vivarium.environments.braitenberg.selective_sensing import AgentState, ObjectState


@pytest.fixture
def scene_config():
    return SceneConfiguration('braitenberg')


@pytest.fixture
def tmp_state(scene_config):
    return scene_config.create_state()


@pytest.fixture
def agent_field(tmp_state):
    return tmp_state.field_name(AgentState)


@pytest.fixture
def object_field(tmp_state):
    return tmp_state.field_name(ObjectState)


@pytest.fixture
def get_point_particle_state(scene_config):
    def _get_state():
        return scene_config.create_state()
    return _get_state


@pytest.fixture
def get_rigid_body_state(get_point_particle_state):
    def _get_state():
        state = get_point_particle_state()
        state = state.set(entity_state=to_rigid_body_state(state.entity_state))
        return state
    return _get_state


def generate_changes(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field):
    return {
        agent_field: {'wheel_diameter': [{'__idx': wheel_diameter_idx, '__value': wheel_diameter_value}]},
        'entity_state': {
            'exists': [{'__idx': exists_idx, '__value': exists_value}],
            'friction': [{'__idx': friction_idx, '__value': friction_value},
                         {'__idx': slice(friction_idx, friction_idx + 2), '__value': friction_value + 1}]
        }
    }


def generate_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field):
    return {
        agent_field: {'wheel_diameter': lambda state: state.field(AgentState).wheel_diameter.at[wheel_diameter_idx].set(wheel_diameter_value)},
        'entity_state': {
            'exists': lambda state: state.entity_state.exists.at[exists_idx].set(exists_value),
            'friction': lambda state: state.entity_state.friction.at[friction_idx].set(friction_value).at[friction_idx:friction_idx+2].set(friction_value + 1)
        }
    }


def generate_changes_and_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field):
    changes = generate_changes(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field)
    expected = generate_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field)
    return changes, expected


@pytest.mark.parametrize("changes_and_expected, state_fn", [
    (lambda agent_field: generate_changes_and_expected(0, 0, 7, 0, 6, 0, agent_field), "get_rigid_body_state"),
    (lambda agent_field: generate_changes_and_expected(1, 1, 8, 1, 7, 1, agent_field), "get_point_particle_state"),
])
def test_change_recorder(changes_and_expected, state_fn, request, agent_field):
    state_fn = request.getfixturevalue(state_fn)
    state = state_fn()

    changes, expected = changes_and_expected(agent_field)

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

    state = update_dataclass(state, fetched_changes)

    for entity, attrs in expected.items():
        for attr, expected_fn in attrs.items():
            assert (jnp.equal(getattr(getattr(state, entity), attr), expected_fn(state))).all()


def test_dataclass_wrapper(get_rigid_body_state):
    state = get_rigid_body_state()
    idx = 3
    val = [0.2, 0.3]
    agent_field = state.field_name(AgentState)
    cur_val = state.field(AgentState).motor[idx]
    assert (not jnp.equal(jnp.array(cur_val), jnp.array(val)).all())

    state = getattr(DataclassWrapper(), agent_field).motor[idx].set(val).apply(state)

    assert (jnp.equal(jnp.array(state.field(AgentState).motor[idx]), jnp.array(val)).all())


def test_dataclass_wrapper_with_state(get_rigid_body_state):
    state = get_rigid_body_state()
    agent_field = state.field_name(AgentState)
    dw = DataclassWrapper(state)
    
    assert jnp.equal(state.entity_state.position.center, dw.entity_state.position.center).all()

    getattr(dw, agent_field).motor = 100 * jnp.ones_like(state.field(AgentState).motor)
    
    dw.apply()

    assert jnp.equal(getattr(dw, agent_field).motor, 100 * jnp.ones_like(state.field(AgentState).motor)).all()  
    
    assert jnp.equal(state.entity_state.position.center[1, 2], dw.entity_state.position.center[1, 2]).all()


@pytest.mark.parametrize("idx, position_center, position_orientation, init_state_fn, entity_type", [
    (2, [7, 8], 2.0, "get_rigid_body_state", "agent_field"),
    (3, [5, 6], 1.5, "get_point_particle_state", "object_field"),
    (4, [9, 10], 3.0, "get_rigid_body_state", "agent_field"),
])
def test_entity_wrapper(idx, position_center, position_orientation, init_state_fn, entity_type, request):
    state_fn = request.getfixturevalue(init_state_fn)
    entity_type = request.getfixturevalue(entity_type)
    state = state_fn()

    entity = EntityWrapper(state, idx, entity_type)

    entity.position_center = position_center
    entity.position_orientation = position_orientation

    previous_state = state

    state = entity.apply_to_state(state)

    assert (not (jnp.equal(previous_state.entity_state.position_center[idx], jnp.array(position_center))).all())
    assert (jnp.equal(state.entity_state.position_center[idx], jnp.array(position_center))).all()
    assert (jnp.equal(state.entity_state.position_center, previous_state.entity_state.position_center.at[idx].set(position_center))).all()
    assert (jnp.equal(state.entity_state.position_orientation, previous_state.entity_state.position_orientation.at[idx].set(position_orientation))).all()


@pytest.mark.parametrize("idx, position_center, entity_type, state_fn", [
    (4, [7, 8], "agent_field", "get_rigid_body_state"),
    (2, [9, 10], "object_field", "get_point_particle_state"),
])
def test_entity_list(idx, position_center, entity_type, state_fn, request):
    state_fn = request.getfixturevalue(state_fn)
    entity_type = request.getfixturevalue(entity_type)
    scene_config = request.getfixturevalue("scene_config")
    state = state_fn()
    
    objects = EntityList(state, entity_type, scene_config.entity_type_configs[entity_type].idx)

    obj = objects[idx]
    obj.position_center = position_center

    state = objects.apply_to_state(state)

    entity_state = getattr(state, entity_type)
    expected_position = state.entity_state.position_center.at[entity_state.entity_idx[idx]].set(position_center)
    
    state = objects.apply_to_state(state)
    entity_state = getattr(state, entity_type)

    assert (jnp.equal(state.entity_state.position_center, expected_position)).all()


def test_on_simulator_instance(scene_config):
    simulator = scene_config.create_simulator()
    dw = DataclassWrapper()
    dw.freq = 42
    simulator = dw.apply(simulator)
    assert simulator.freq == 42

    dw = DataclassWrapper()
    dw.env.num_scan_steps = 42
    simulator = dw.apply(simulator)
    assert simulator.env.num_scan_steps == 42


def test_simulator_apply_change(scene_config):
    simulator = scene_config.create_simulator()
    dw = DataclassWrapper()
    dw.box_size = 42.
    dw.freq = -10.
    changes = dw.fetch_changes()
    simulator.apply_changes([changes])
    assert simulator.env.box_size == 42.
    assert simulator.freq == -10.
