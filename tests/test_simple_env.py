from vivarium.environments.braitenberg.simple.simple_env import (
    init_state,
    BraitenbergEnv,
)

NUM_STEPS = 10


def test_simple_env_running():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state()
    env = BraitenbergEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state)

    assert env
    assert state
