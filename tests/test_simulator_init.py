from vivarium.environments.braitenberg.selective_sensing import (
    init_state,
    SelectiveSensorsEnv,
)

from vivarium.simulator.simulator import Simulator


def test_init_simulator_no_args():
    """Test the initialization of the simulator  without arguments"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)

    assert simulator
