import pytest
import jax.numpy as jnp

from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.simulator import Simulator


@pytest.mark.parametrize('scene_name, entity_type',
                         [('braitenberg', 'agents'), 
                          ('particle_lenia', 'particles')
                          ])
def test_state(scene_name, entity_type):
    config = SceneConfiguration(scene_name)
    env = config.create_environment()
    state = env.init_state()
    n_entities = getattr(config.config.environment.components, entity_type).n_max
    assert state.entity_state.exists[:n_entities].sum() == getattr(config.config.environment.components, entity_type).n_exists
    entity_pos = getattr(config.config.environment.components, entity_type).position
    assert jnp.equal(state.entity_state.position[:n_entities], jnp.array(entity_pos)).all()
    if entity_type == 'agents':
        assert state.agents.prox.shape == (n_entities, 2)


def test_component_factories():
    scene_config = SceneConfiguration('braitenberg')
    scene_config.config.environment.kwargs['to_jit'] = False
    env = scene_config.create_environment()
    state = env.init_state()
    idx = 1
    state = state.set(
        agents=state.agents.set(
            motor=state.agents.motor.at[idx, :].set([1., 1.]),
            behavior=state.agents.behavior.at[idx, 0].set(5)
        )
    )
    for t in range(2):
        p = state.entity_state.position[idx]
        state = env.step(state, scan=False)   
    pass


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
    assert simulator.freq == config.config.simulator.kwargs.freq


def test_controller_parameters():
    config = SceneConfiguration('braitenberg')

    #TODO: not ideal, see if how to avoid this
    config.config.environment.components.agents = config.compute_parameters('agents', config.config.environment.components.agents)
    
    controller_parameters = config.create_controller_parameters()
    assert controller_parameters.agents.color == ['red'] * config.config.environment.components.agents.n_max


def test_simulator_controller():
    scene_config = SceneConfiguration('braitenberg')

    env = scene_config.create_environment()
    simulator = Simulator(env=env, scene_name=scene_config.scene_name)
    controller = SimulatorController.from_config(scene_config=scene_config, client=simulator)
    pass