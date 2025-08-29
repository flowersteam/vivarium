import pytest
from typing import List
from dataclasses import dataclass

import jax.numpy as jnp

from vivarium.controllers.dataclass_wrapper import DataclassWrapper
from vivarium.simulator.grpc_server.converters import dataclass_to_proto, proto_to_dataclass, changes_to_proto, proto_to_changes


scene_name = 'braitenberg'


@pytest.fixture
def state(environment_from_config):
    env = environment_from_config(scene_name)
    return env.init_state()


@pytest.fixture
def simulator(simulator_from_config):
    return simulator_from_config(scene_name)


def test_state_de_serialization(state):
    p_state = dataclass_to_proto(state)
    state_2 = proto_to_dataclass(p_state, state.__class__)
    assert state.entity_state.position_center[0, 1] == state_2.entity_state.position_center[0, 1]


def test_parameters_de_serialization(simulator):
    simulator_parameters = simulator.get_simulator_parameters()
    p_parameters = dataclass_to_proto(simulator_parameters)
    simulator_parameters_2 = proto_to_dataclass(p_parameters)

    assert simulator_parameters.freq == simulator_parameters_2.freq
    assert simulator_parameters.box_size == simulator_parameters_2.box_size


def test_changes(state):
    dw = DataclassWrapper()
    idx = (1, 0)
    value = 42.0
    dw.entity_state.position[idx] = value
    other_dim_value = state.entity_state.position[idx[0]][1]
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = dw.update_dataclass(state, changes_2)
    assert jnp.equal(jnp.array(value), state.entity_state.position.__getitem__(idx)).all()
    assert other_dim_value == state.entity_state.position[idx[0]][1]

    dw.entity_state.position[0, 0:2] = [1., 2.]
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = dw.update_dataclass(state, changes_2)
    assert jnp.equal(jnp.array([1., 2.]), state.entity_state.position[0]).all()

    dw.entity_state.position[0:2, 0] = 43.0
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = dw.update_dataclass(state, changes_2)
    assert jnp.equal(jnp.array([43.0, 43.0]), state.entity_state.position[0:2, 0]).all()
    

def test_simulator_grpc(simulator):
    dw = DataclassWrapper()
    dw.env.num_scan_steps = 42
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator2 = dw.update_dataclass(simulator, changes_2)
    assert simulator2 is simulator
    assert simulator2.env.num_scan_steps == 42

    dw.state.entity_state.friction = 42 * jnp.ones_like(simulator.state.entity_state.friction)
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator = dw.update_dataclass(simulator, changes_2)
    assert jnp.equal(jnp.array(42), simulator.state.entity_state.friction).all()


def test_index_grpc():
    @dataclass
    class Test:
        visible_wheels: List
        def set(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    test = Test([True, True])
    dw = DataclassWrapper()
    dw.visible_wheels[0] = False
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    dw.update_dataclass(test, changes_2)
    assert test.visible_wheels == [False, True]
    
    
def test_controller_parameters(simulator):
    cp = simulator.controller_parameters
    p_cp = dataclass_to_proto(cp)
    cp_2 = proto_to_dataclass(p_cp)

    assert cp.agents.color[1] == cp_2.agents.color[1]
