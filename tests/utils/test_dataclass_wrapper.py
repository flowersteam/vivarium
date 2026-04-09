import pytest
import jax.numpy as jnp

from vivarium.utils.dataclass_wrapper import *
from vivarium.components.entities.controller import EntityList, EntityWrapper


scene_name = 'braitenberg'


@pytest.fixture
def init_state(environment_from_config):
    env = environment_from_config(scene_name)
    return env.init_state()


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
        agent_field: {'wheel_diameter': lambda state: getattr(state, agent_field).wheel_diameter.at[wheel_diameter_idx].set(wheel_diameter_value)},
        'entity_state': {
            'exists': lambda state: state.entity_state.exists.at[exists_idx].set(exists_value),
            'friction': lambda state: state.entity_state.friction.at[friction_idx].set(friction_value).at[friction_idx:friction_idx+2].set(friction_value + 1)
        }
    }


def generate_changes_and_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field):
    changes = generate_changes(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field)
    expected = generate_expected(wheel_diameter_idx, wheel_diameter_value, exists_idx, exists_value, friction_idx, friction_value, agent_field)
    return changes, expected


@pytest.mark.parametrize("changes_and_expected", [
    lambda agent_field: generate_changes_and_expected(0, 0, 7, False, 6, 0, agent_field),
    lambda agent_field: generate_changes_and_expected(1, 1, 8, True, 7, 1, agent_field),
])
def test_remote_fetch_and_update(changes_and_expected, init_state):
    state = init_state

    changes, expected = changes_and_expected('agents')

    remote = Remote()

    for entity, attrs in changes.items():
        for attr, changes_list in attrs.items():
            for change in changes_list:
                if change['__idx'] is None:
                    setattr(getattr(remote, entity), attr, change['__value'])
                else:
                    getattr(getattr(remote, entity), attr)[change['__idx']] = change['__value']

    fetched_changes = remote.fetch_changes()[0]

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


def test_remote_apply(init_state):
    state = init_state
    idx = 3
    val = [0.2, 0.3]
    cur_val = state.agents.motor[idx]
    assert (not jnp.equal(jnp.array(cur_val), jnp.array(val)).all())

    remote = Remote()
    remote.agents.motor[idx] = val
    state = remote.apply(state)

    assert (jnp.equal(jnp.array(state.agents.motor[idx]), jnp.array(val)).all())


def test_remote_with_state(init_state):
    state = init_state
    dw = Remote(state, set_obj=True)

    assert jnp.equal(state.entity_state.position, dw.entity_state.position.obj()).all()

    dw.agents.motor = 100 * jnp.ones_like(state.agents.motor)

    dw.apply()

    assert jnp.equal(dw.agents.motor.obj(), 100 * jnp.ones_like(state.agents.motor)).all()

    assert jnp.equal(state.entity_state.position[1, 2], dw.entity_state.position[1, 2]).all()


def test_on_simulator_instance(simulator_from_config):
    simulator = simulator_from_config(scene_name)
    remote = Remote()
    remote.controller_parameters.simulator.freq = 42
    simulator = remote.apply(simulator)
    assert simulator.controller_parameters.simulator.freq == 42

    remote = Remote()
    remote.env.num_scan_steps = 42
    simulator = remote.apply(simulator)
    assert simulator.env.num_scan_steps == 42


def test_simulator_apply_change(simulator_from_config):
    simulator = simulator_from_config(scene_name)
    remote = Remote(simulator)
    remote.controller_parameters.simulator.env.box_size = 42.
    remote.controller_parameters.simulator.freq = -10.
    changes = remote.fetch_changes()
    simulator.set_changes(changes)
    assert simulator.env.box_size == 42.
    assert simulator.controller_parameters.simulator.freq == -10.
