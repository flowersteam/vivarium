import jax.numpy as jnp

from vivarium.environment.components.entities.braitenberg import BraitenbergComponent
from vivarium.environment.components.entities.objects import ObjectComponent

def expected_values_from_config(config):

    agent = config.environment.components.agents
    object = config.environment.components.objects
    n_agents = agent.n_max
    n_objects = object.n_max

    expected_values = {
        "entity_type": [0] * n_agents + [1] * n_objects,
        "entity_type_idx": list(range(n_agents)) + list(range(n_objects)),
        "mass": [[m] for m in agent.mass] + [[m] for m in object.mass],
        "position": agent.position + object.position,
        "force": [[0, 0]] * (n_agents + n_objects),
        "proxs_dist_max": agent.proxs_dist_max
    }

    return expected_values


def test_create_state(scene_config, environment_and_state):
    config = scene_config('braitenberg')
    factories = [BraitenbergComponent.from_config(config.environment.components.agents, name='agents'),
                 ObjectComponent.from_config(config.environment.components.objects, name='objects')]
    _, state = environment_and_state(factories)

    expected_values = expected_values_from_config(config)

    assert jnp.equal(state.agents.proxs_dist_max, jnp.array(expected_values["proxs_dist_max"])).all()
    assert jnp.equal(state.entity_state.entity_type, jnp.array(expected_values["entity_type"])).all()
    assert jnp.equal(state.entity_state.entity_type_idx, jnp.array(expected_values["entity_type_idx"])).all()
    assert jnp.equal(state.entity_state.mass, jnp.array(expected_values["mass"])).all()
    assert jnp.equal(state.entity_state.position, jnp.array(expected_values["position"])).all()
    assert jnp.equal(state.entity_state.force, jnp.array(expected_values["force"])).all()
