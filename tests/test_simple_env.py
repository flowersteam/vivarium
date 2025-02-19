
from vivarium.environments.braitenberg.simple import (
    init_state,
    BraitenbergEnv,
)

from vivarium.environments.braitenberg import simple
from vivarium.environments.utils import rigid_body_to_point_particle

init_state_point_particle, _ = rigid_body_to_point_particle(simple)

NUM_STEPS = 10


def test_simple_env_rigid_body_running():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state()
    env = BraitenbergEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state, num_scan_steps=3)

    assert env
    assert state
    assert state.entities.is_rigid_body()


def test_simple_env_point_particle_running():
    """Test the stepping mechanism of the env with occlusion (default)"""
    state = init_state_point_particle()
    env = BraitenbergEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state, num_scan_steps=3)

    assert env
    assert state
    assert not state.entities.is_rigid_body()
