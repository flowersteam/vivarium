import pytest

import jax.numpy as jnp

from vivarium.environments.objects.state import ObjectState
from vivarium.utils.scene_configs import SceneConfiguration

from vivarium.environments.state import EntityState

from vivarium.environments.braitenberg.selective_sensing.state import AgentState


scene_config = SceneConfiguration('braitenberg')

state = scene_config.create_state()
agent_field = state.field_name(AgentState)
object_field = state.field_name(ObjectState)

agent = scene_config.entity_type_configs[agent_field].kwargs
object = scene_config.entity_type_configs[object_field].kwargs
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

entity_types_kwargs = {etype: config.kwargs for etype, config in scene_config.entity_type_configs.items()}


@pytest.mark.parametrize("entity_type, entity_idx_offset", [
    (agent_field, 0),
    (object_field, 2),
])
def test_create_entity_type_state(entity_type, entity_idx_offset):
    cls = scene_config.entity_type_configs[entity_type].state_cls
    state = cls.create(entity_idx_offset, entity_types_kwargs, entity_type)
    n_exists = n_agents if entity_type == agent_field else n_objects
    assert jnp.equal(state.entity_idx, jnp.array(range(entity_idx_offset, entity_idx_offset + n_exists))).all()

    if state.__class__ == AgentState:
        assert jnp.equal(state.proxs_dist_max, jnp.array(expected_values["proxs_dist_max"])).all()


def test_create_entity_state():

    
    entity_state = EntityState.create(scene_config.entity_types, entity_types_kwargs)
    
    assert jnp.equal(entity_state.entity_type, jnp.array(expected_values["entity_type"])).all()
    assert jnp.equal(entity_state.entity_type_idx, jnp.array(expected_values["entity_type_idx"])).all()
    assert jnp.equal(entity_state.mass, jnp.array(expected_values["mass"])).all()
    assert jnp.equal(entity_state.position, jnp.array(expected_values["position"])).all()
    assert jnp.equal(entity_state.force, jnp.array(expected_values["force"])).all()


def test_create_state():

    state = scene_config.create_state()

    assert jnp.equal(state.entity_state.entity_type, jnp.array(expected_values['entity_type'])).all()
    assert jnp.equal(state.entity_state.entity_type_idx, jnp.array(expected_values['entity_type_idx'])).all()
    assert jnp.equal(state.entity_state.mass, jnp.array(expected_values['mass'])).all()
    assert jnp.equal(state.entity_state.position, jnp.array(expected_values['position'])).all()
