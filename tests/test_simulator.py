import pytest

from vivarium.simulator import Simulator


NUM_STEPS = 6


@pytest.mark.parametrize("scene_name", ["braitenberg", "particle_lenia", "lenia_braitenberg"])
def test_simulator_run(scene_name, simulator_from_config):
    simulator = simulator_from_config(scene_name)
    for _ in range(NUM_STEPS):
        simulator.step()

    assert simulator


def test_load_save_simulator_config(scene_config):
    config = scene_config('braitenberg')
    simulator = Simulator.from_config(config.simulator)

    state = simulator.env.init_state()
    state = simulator.env.step(state)

    simulator.freq = 42.

    updated_config = simulator.to_config(state)

    assert updated_config.freq == 42.

    new_simulator = Simulator.from_config(updated_config)

    assert new_simulator.freq == 42.
    assert new_simulator

    pass
