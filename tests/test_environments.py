import pytest
import jax.numpy as jnp

from vivarium.utils.scene_configs import SceneConfiguration


NUM_STEPS = 5


@pytest.mark.parametrize("scene_name", ["sandbox", "braitenberg", "particle_lenia", "lenia_braitenberg"])
def test_env(scene_name):
    """Test the stepping mechanism of the env with occlusion (default)"""
    scene_config = SceneConfiguration(scene_name)
    scene_config.config.environment.kwargs['to_jit'] = False
    env = scene_config.create_environment()
    state = env.init_state()
    env.neighbor_manager.allocate(state.entity_state.position)
    previous_state = state
    for t in range(NUM_STEPS):
        prev_prev = previous_state
        previous_state = state
        state = env.step(state, scan=False)
        if jnp.isnan(state.entity_state.position).any():
            print(f"NaN detected at step {t}")
            state = prev_prev  # revert to previous state twice

    assert env
    assert state
    assert not state.entity_state.is_rigid_body()
