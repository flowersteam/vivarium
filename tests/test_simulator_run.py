from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)
from vivarium.simulator.simulator import Simulator

NUM_STEPS = 50


def test_simulator_run():
    """Test default simulator run"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)

    assert simulator
