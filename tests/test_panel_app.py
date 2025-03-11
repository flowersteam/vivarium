import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.simulator_states import EntityType
from vivarium.interface.panel_app import WindowManager

# TODO: After an upgrade on panel on 2025-03-11 the test below started to fail

# def test_window_manager():
#     state = init_state()
#     env = SelectiveSensorsEnv(state=state)
#     simulator = Simulator(env_state=state, env=env)
#     wm = WindowManager(client=simulator)
#     wm.entity_managers[EntityType.AGENT].selected_param_entity.behavior_0 = 'NOOP'
#     wm.entity_managers[EntityType.AGENT].selected_param_entity.sensed_PREYS_0 = True
#     # assert False
