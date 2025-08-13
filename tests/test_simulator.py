import pytest

from vivarium.simulator import Simulator
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.environments.entities.braitenberg.component import AgentState
from vivarium.environments.entities.particle_lenia.component import ParticleLeniaState

NUM_STEPS = 6


@pytest.mark.parametrize("scene_name", ["braitenberg", "particle_lenia", "lenia_braitenberg"])
def test_simulator_run(scene_name, simulator_from_config):
    simulator = simulator_from_config(scene_name)
    for _ in range(NUM_STEPS):
        simulator.step()

    assert simulator

def test_load_simulator_config():
    scene_config = SceneConfiguration('braitenberg')
    simulator_config = scene_config.config.simulator
    simulator = Simulator.from_config(simulator_config)

    state = simulator.env.init_state()
    state = simulator.env.step(state)

    simulator.freq = 42.

    updated_config = simulator.to_config(state)

    assert updated_config.freq == 42.

    new_simulator = Simulator.from_config(updated_config)

    assert new_simulator.freq == 42.
    assert new_simulator

    pass
