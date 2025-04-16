import jax.numpy as jnp

from vivarium.utils.scene_configs import SceneConfiguration
import pytest

@pytest.mark.parametrize('scene_name, entity_type',
                         [('braitenberg', 'agents'), 
                          ('particle_lenia', 'particles')
                          ])
def test_state(scene_name, entity_type):
    config = SceneConfiguration(scene_name)
    state = config.create_state()
    n_entities = getattr(config.config.entities, entity_type).kwargs.n_exists
    entity_pos = getattr(config.config.entities, entity_type).kwargs.position
    assert jnp.equal(state.entity_state.position[:n_entities], jnp.array(entity_pos)).all()
    if entity_type == 'agents':
        assert state.agents.prox.shape == (n_entities, 2)

@pytest.mark.parametrize('scene_name', ['braitenberg', 'particle_lenia'])
def test_environment(scene_name):
    config = SceneConfiguration(scene_name)
    env = config.create_environment()
    assert env.box_size == config.config.environment.kwargs.box_size

@pytest.mark.parametrize('scene_name', ['braitenberg', 'particle_lenia'])
def test_simulator(scene_name):
    config = SceneConfiguration(scene_name)
    simulator = config.create_simulator()
    assert simulator.env.box_size == config.config.environment.kwargs.box_size
    assert simulator.freq == config.config.simulator.kwargs.update_freq
