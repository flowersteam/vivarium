from jax import random
import jax.numpy as jnp

from vivarium.simulator.grpc_server.converters import state_to_proto, proto_to_state, changes_to_proto, proto_to_changes
from vivarium.controllers.dataclass_wrapper import DataclassWrapper
from vivarium.environments.physics_engine import init_state_fn
from vivarium.utils.scene_configs import load_scene_config
from vivarium.environments.base_env import create_state
from vivarium.simulator.simulator import Simulator


def get_state():
    state = create_state(load_scene_config('braitenberg'))
    state = init_state_fn(random.PRNGKey(0))(state)
    return state


def test_de_serialiazation():
    state = get_state()
    p_state = state_to_proto(state)
    state_2 = proto_to_state(p_state, state.__class__)
    assert state.entity_state.position_center[0, 1] == state_2.entity_state.position_center[0, 1]


def test_changes():
    state = get_state()
    dw = DataclassWrapper()
    idx = (1, 0)
    value = 42.
    dw.entity_state.position[idx] = value
    other_dim_value = state.entity_state.position[idx[0]][1]
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state  = dw.update_state(state, changes_2)
    assert jnp.equal(jnp.array(value),
                               state.entity_state.position.__getitem__(idx)).all()

    assert other_dim_value == state.entity_state.position[idx[0]][1]


def test_simulator_grpc():
    simulator = Simulator.from_scene('braitenberg')

    dw = DataclassWrapper()
    dw.env.num_scan_steps = 42
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator2 = dw.update_state(simulator, changes_2)
    assert simulator2 is simulator
    assert simulator2.env.num_scan_steps == 42

    # dw. = DataclassWrapper()
    dw.state.entity_state.friction = 42 * jnp.ones_like(simulator.state.entity_state.friction)
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    simulator  = dw.update_state(simulator, changes_2)
    assert jnp.equal(jnp.array(42), simulator.state.entity_state.friction).all()
