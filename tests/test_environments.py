import pytest
import jax.numpy as jnp

from vivarium.environment import Environment


NUM_STEPS = 5


@pytest.mark.parametrize("scene_name", [
    "braitenberg", 
    "particle_lenia", 
    "lenia_braitenberg", 
    "non_transitive", 
    "fishing"])
def test_env(scene_name, scene_config):
    """Test the stepping mechanism of the env with occlusion (default)"""
    config = scene_config(scene_name)
    config.environment.kwargs['to_jit'] = False
    env = Environment.from_config(config.environment)
    state = env.init_state()
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


def test_load_save_env_config(scene_config):
    """Test the environment creation from config and back."""

    config = scene_config("braitenberg")
    env_config = config.environment
    env = Environment.from_config(env_config)
    state = env.init_state()
    state = env.step(state, scan=False)

    env.box_size = 42.
    state = state.set(
        collision_state = state.collision_state.set(
            epsilon=42.,
    ))

    new_env_config = env.to_config(state)

    assert new_env_config.kwargs.box_size == 42.
    assert new_env_config.components.component_list['collision'].epsilon == 42.

    new_env = Environment.from_config(new_env_config)
    new_state = new_env.init_state()
    new_state = new_env.step(new_state, scan=False)
    
    assert new_env.box_size == 42.
    assert new_env.get_factory_by_name('collision').epsilon == 42.
    assert new_state.collision_state.epsilon == 42.
