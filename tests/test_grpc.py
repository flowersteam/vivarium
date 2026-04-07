import pytest
from typing import List
from dataclasses import dataclass

import jax.numpy as jnp

from vivarium.utils.dataclass_wrapper import Remote
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


def test_parameters_de_serialization(simulator_from_config):
    simulator = simulator_from_config(scene_name)
    simulator_parameters = simulator.controller_parameters.simulator
    p_parameters = dataclass_to_proto(simulator_parameters)
    simulator_parameters_2 = proto_to_dataclass(p_parameters)

    assert simulator_parameters.freq == simulator_parameters_2.freq
    assert simulator_parameters.env.box_size == simulator_parameters_2.env.box_size


def test_changes(state):
    remote = Remote()
    idx = (1, 0)
    value = 42.0
    remote.entity_state.position[idx] = value
    other_dim_value = state.entity_state.position[idx[0]][1]
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = remote.apply(state, changes_2)
    assert jnp.equal(jnp.array(value), state.entity_state.position.__getitem__(idx)).all()
    assert other_dim_value == state.entity_state.position[idx[0]][1]

    remote.entity_state.position[0, 0:2] = [1., 2.]
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = remote.apply(state, changes_2)
    assert jnp.equal(jnp.array([1., 2.]), state.entity_state.position[0]).all()

    remote.entity_state.position[0:2, 0] = 43.0
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state = remote.apply(state, changes_2)
    assert jnp.equal(jnp.array([43.0, 43.0]), state.entity_state.position[0:2, 0]).all()
    

def test_simulator_grpc(simulator):
    remote = Remote()
    remote.env.num_scan_steps = 42
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator2 = remote.apply(simulator, changes_2)
    assert simulator2 is simulator
    assert simulator2.env.num_scan_steps == 42

    remote.state.entity_state.friction = 42 * jnp.ones_like(simulator.state.entity_state.friction)
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator = remote.apply(simulator, changes_2)
    assert jnp.equal(jnp.array(42), simulator.state.entity_state.friction).all()


def test_index_grpc():
    @dataclass
    class Test:
        visible_wheels: List
        def set(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    test = Test([True, True])
    remote = Remote()
    remote.visible_wheels[0] = False
    changes = remote.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    remote.apply(test, changes_2)
    assert test.visible_wheels == [False, True]
    
    
def test_controller_parameters(simulator):
    cp = simulator.controller_parameters
    p_cp = dataclass_to_proto(cp)
    cp_2 = proto_to_dataclass(p_cp)

    assert cp.agents.color[1] == cp_2.agents.color[1]


@pytest.mark.slow
def test_bidirectional_streaming(grpc_client):
    """Test bidirectional streaming RPC."""
    client = grpc_client(scene_name)
    
    num_steps = 5
    received_states = []
    
    def changes_generator():
        for i in range(num_steps):
            yield []  # No changes, just step
    
    for state_and_cp in client.bidirectional_step_generator(changes_generator()):
        received_states.append(state_and_cp)
    
    # Should receive one state per step
    assert len(received_states) == num_steps
    
    # Each state should be valid
    for state_and_cp in received_states:
        assert state_and_cp.state is not None
        assert state_and_cp.controller_parameters is not None
    

@pytest.mark.slow
def test_set_changes(grpc_client):
    """Test RPC for streaming mode."""
    client = grpc_client(scene_name)
    
    # Verify is_streaming is False initially
    assert not client.is_streaming
    
    # Apply changes with update_from_server=False
    # (doesn't update local state, just sends to server)
    initial_state = client.state
    client.set_changes([], update_from_server=False)
    
    # Local state should still be the same reference
    # (set_changes doesn't update it)
    assert client.state is initial_state
    
    # But with update_from_server=True should update it
    client.set_changes([], update_from_server=True)
    assert client.state is not initial_state
    