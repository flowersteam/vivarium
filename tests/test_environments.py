from vivarium.utils.scene_configs import SceneConfiguration
import pytest

NUM_STEPS = 10


@pytest.mark.parametrize("scene_name", ["braitenberg", "particle_lenia", "lenia_braitenberg"])
def test_env(scene_name):
    """Test the stepping mechanism of the env with occlusion (default)"""
    scene_config = SceneConfiguration(scene_name)
    scene_config.config.environment.kwargs['to_jit'] = False
    env = scene_config.create_environment()
    state = env.state
    for _ in range(NUM_STEPS):
        state = env.step(state)

    assert env
    assert state
    assert not state.entity_state.is_rigid_body()

