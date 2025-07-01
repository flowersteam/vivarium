import pytest
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
