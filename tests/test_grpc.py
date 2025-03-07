from jax import random
import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing import init_state
from vivarium.simulator.simulator import env_to_sim_state
from vivarium.environments.physics_engine import init_state_fn
from vivarium.simulator.grpc_server.converters import state_to_proto, proto_to_state, changes_to_proto, proto_to_changes
from vivarium.controllers.dataclass_wrapper import DataclassWrapper
import dataclasses


def get_state():
    state = init_state()
    state = env_to_sim_state(state, num_steps_lax=3, freq=60., 
                                 use_fori_loop=False, to_jit=True)
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
    dw.entity_state.position.center[idx] = value
    other_dim_value = state.entity_state.position.center[idx[0]][1]
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state  = dw.update_state(state, changes_2)
    assert jnp.equal(jnp.array(value),
                               state.entity_state.position.center.__getitem__(idx)).all()

    assert other_dim_value == state.entity_state.position.center[idx[0]][1]

    dw = DataclassWrapper()
    dw.simulator_state.box_size = 42.
    changes = dw.fetch_changes()
    p_changes = changes_to_proto(changes)
    changes_2 = proto_to_changes(p_changes)
    state  = dw.update_state(state, changes_2)
    assert jnp.equal(jnp.array(42.),
                               state.simulator_state.box_size).all()
