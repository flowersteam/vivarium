from vivarium.environments.braitenberg.selective_sensing import (
    init_state as init_state_rigid_body,
    SelectiveSensorsEnv,
)

from vivarium.environments.braitenberg import selective_sensing
from vivarium.environments.utils import rigid_body_to_point_particle

init_state_point_particle, _ = rigid_body_to_point_particle(selective_sensing)

NUM_STEPS = 10


def test_env_running_rigid_body():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state_rigid_body()
    env = SelectiveSensorsEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state, num_scan_steps=3)

    assert env
    assert state
    assert state.entities.is_rigid_body()

def test_env_running_point_particle():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state_point_particle()
    env = SelectiveSensorsEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state, num_scan_steps=3)

    assert env
    assert state
    assert not state.entities.is_rigid_body()

# def test_env_running_no_occlusion():
#     """Test the stepping mechanism of the env without occlusion"""
#     state = init_state()
#     env = SelectiveSensorsEnv(state=state, occlusion=False)

#     for _ in range(NUM_STEPS):
#         state = env.step(state=state)

#     assert env
#     assert state
