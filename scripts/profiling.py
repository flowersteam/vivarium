from vivarium.utils.scene_configs import load_scene_config
from vivarium.environment import Environment
from jax import profiler

config = load_scene_config('braitenberg')
env = Environment.from_config(config.environment)
state = env.init_state()

with profiler.trace("/tmp/tensorboard"):
    env.step(state).entity_state.position.block_until_ready()
