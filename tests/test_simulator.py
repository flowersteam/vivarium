import pytest
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.environments.entities.braitenberg import AgentState
from vivarium.environments.entities.particle_lenia import ParticleLeniaState

NUM_STEPS = 10


@pytest.mark.parametrize("scene_name", ["braitenberg", "particle_lenia"])
def test_simulator_run(scene_name):
    simulator = SceneConfiguration(scene_name).create_simulator()
    for _ in range(NUM_STEPS):
        simulator.step()

    assert simulator


def test_load_scene():
    simulator = SceneConfiguration('braitenberg').create_simulator()
    assert simulator.state.field(AgentState).__class__ == AgentState
    with pytest.raises(ValueError):
        simulator.state.field(ParticleLeniaState)

    simulator.load_scene('particle_lenia')
    with pytest.raises(ValueError):
        simulator.state.field(AgentState) is None
    assert simulator.state.field(ParticleLeniaState).__class__ == ParticleLeniaState