from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)

NUM_STEPS = 10


def test_env_running_occlusion():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state, occlusion=True)

    for _ in range(NUM_STEPS):
        state = env.step(state=state)

    assert env
    assert state


def test_env_running_no_occlusion():
    """Test the stepping mechanism of the env without occlusion"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state, occlusion=False)

    for _ in range(NUM_STEPS):
        state = env.step(state=state)

    assert env
    assert state
