from vivarium.simulator.simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)


def test_init_simulator_no_args():
    """Test the initialization of the simulator  without arguments"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)

    assert simulator
