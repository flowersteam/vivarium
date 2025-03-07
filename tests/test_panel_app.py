import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)
from vivarium.simulator.simulator import Simulator


from vivarium.interface.panel_app import WindowManager


def test_window_manager():
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)
    wm = WindowManager(client=simulator)
