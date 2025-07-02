from vivarium.utils.scene_configs import SceneConfiguration
from jax import profiler


scene_config = SceneConfiguration('braitenberg')
env = scene_config.create_environment()

state = env.init_state()

num_steps = 1

with profiler.trace("/tmp/tensorboard"):  #"/tmp/jax-trace", create_perfetto_link=True):
    env._step_env(state, env.neighbor_manager.neighbors, num_steps)[0].entity_state.position.block_until_ready()
