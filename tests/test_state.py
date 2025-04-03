import pytest

import jax.numpy as jnp

from vivarium.utils.scene_configs import load_scene_config

from vivarium.environments.state import (
    EntityState, 
    create_state,
    class_to_string
    )

from vivarium.environments.braitenberg.selective_sensing.state import (
    AgentState,
    ObjectState
    )

config = load_scene_config('braitenberg')

agent = config.state_data.agent_state.kwargs
object = config.state_data.object_state.kwargs
n_agents = agent.n_exists
n_objects = object.n_exists

expected_values = {
    "entity_type": [0] * n_agents + [1] * n_objects,
    "entity_type_idx": list(range(n_agents)) + list(range(n_objects)),
    "mass": [[m] for m in agent.mass] + [[m] for m in object.mass],
    "position": agent.position + object.position,
    "force": [[0, 0]] * (n_agents + n_objects),
    "proxs_dist_max": agent.proxs_dist_max
}

def test_class_to_string():
    assert class_to_string(AgentState) == 'agent_state'
    assert class_to_string(ObjectState) == 'object_state'


@pytest.mark.parametrize("entity_type_cls, entity_idx_offset", [
    (AgentState, 0),
    (ObjectState, 2),
])
def test_create_entity_type_state(entity_type_cls, entity_idx_offset):
    cls = entity_type_cls
    state = cls.create(entity_idx_offset, config, class_to_string(cls))

    assert jnp.equal(state.entity_idx, jnp.array(range(entity_idx_offset, entity_idx_offset + config['state_data'][class_to_string(cls)]['kwargs']['n_exists']))).all()

    if entity_type_cls == AgentState:
        assert jnp.equal(state.proxs_dist_max, jnp.array(expected_values["proxs_dist_max"])).all()


def test_create_entity_state():

    entity_state = EntityState.create(config)
    
    assert jnp.equal(entity_state.entity_type, jnp.array(expected_values["entity_type"])).all()
    assert jnp.equal(entity_state.entity_type_idx, jnp.array(expected_values["entity_type_idx"])).all()
    assert jnp.equal(entity_state.mass, jnp.array(expected_values["mass"])).all()
    assert jnp.equal(entity_state.position, jnp.array(expected_values["position"])).all()
    assert jnp.equal(entity_state.force, jnp.array(expected_values["force"])).all()


def test_create_state():

    state = create_state(config)

    assert jnp.equal(state.entity_state.entity_type, jnp.array(expected_values['entity_type'])).all()
    assert jnp.equal(state.entity_state.entity_type_idx, jnp.array(expected_values['entity_type_idx'])).all()
    assert jnp.equal(state.entity_state.mass, jnp.array(expected_values['mass'])).all()
    assert jnp.equal(state.entity_state.position, jnp.array(expected_values['position'])).all()
