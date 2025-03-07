import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)
from vivarium.simulator.simulator import Simulator
from vivarium.controllers.simulator_controller import EntityType
from vivarium.controllers.panel_controller import PanelController


def test_panel_controller():
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)
    controller = PanelController(client=simulator)

    idx = 3
    pos = controller.state.entity_state.position_center[idx]

    controller.selected[EntityType.AGENT].selection = [idx]

    ag =controller.agents[idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.visible = False

    controller.selected_entities[EntityType.AGENT].y_position = 5.42
    controller.apply_changes()
    controller.update_state()

    assert controller.state.entity_state.position_center[idx][0] == pos[0]
    assert controller.state.entity_state.position_center[idx][1] == 5.42
    assert controller.agents[idx].position_center[1] == 5.42

