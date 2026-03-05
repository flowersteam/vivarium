import pytest
import jax.numpy as jnp

from vivarium.simulator import Simulator
from vivarium.environment import Environment


@pytest.mark.parametrize('scene_name, entity_type',
                         [('braitenberg', 'agents'), 
                          ('particle_lenia', 'particles')
                          ])
def test_state(scene_name, entity_type, scene_config):
    config = scene_config(scene_name)
    env = Environment.from_config(config.environment)
    state = env.init_state()
    n_entities = getattr(config.environment.components.component_list, entity_type).n_max
    assert jnp.equal(state.entity_state.exists[:n_entities], 
                   jnp.array(getattr(config.environment.components.component_list, entity_type).exists)).all()
    entity_pos = getattr(config.environment.components.component_list, entity_type).position
    assert jnp.equal(state.entity_state.position[:n_entities], jnp.array(entity_pos)).all()
    if entity_type == 'braitenberg':
        entity_name = config.environment.components.component_list[entity_type].name
        assert getattr(state, entity_name).prox.shape == (n_entities, 2)


def test_component_factories(scene_config):
    config = scene_config('braitenberg')
    config.environment.kwargs['to_jit'] = False
    env = Environment.from_config(config.environment)
    state = env.init_state()
    idx = 1
    state = state.set(
        agents=state.agents.set(
            motor=state.agents.motor.at[idx, :].set([1., 1.]),
        )
    )
    for t in range(2):
        p = state.entity_state.position[idx]
        state = env.step(state, scan=False)   
    pass


@pytest.mark.parametrize('scene_name', ['braitenberg', 'particle_lenia'])
def test_environment(scene_name, scene_config):
    config = scene_config(scene_name)
    env = Environment.from_config(config.environment)
    assert env.box_size == config.environment.kwargs.box_size


@pytest.mark.parametrize('scene_name', ['braitenberg', 'particle_lenia'])
def test_simulator(scene_name, scene_config):
    config = scene_config(scene_name)
    simulator = Simulator.from_config(config.simulator)
    assert simulator.env.box_size == config.environment.kwargs.box_size
    assert simulator.freq == config.simulator.client.controller_kwargs.freq


def test_controller_parameters(scene_config):
    config = scene_config('braitenberg')

    controller_parameters = Simulator.from_config(config.simulator).controller_parameters

    assert controller_parameters.agents.color == ['purple'] * 5 + ['red'] * 5
