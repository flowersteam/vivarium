import pytest

import jax.numpy as jnp

from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.controllers.panel_controller import PanelController


@pytest.mark.parametrize("scene_name", [
    'lenia_braitenberg',
    'particle_lenia',
    'braitenberg',
])
def test_panel_controller(scene_name):
    config = SceneConfiguration(scene_name)
    controllers = config.create_controllers()
    simulator = config.create_simulator()
    controller = PanelController(client=simulator, subtypes=config.config.subtypes, **controllers)
    for entity_type in controllers.keys():
        entity_idx = 1
        idx = getattr(controller.state, entity_type).entity_idx[entity_idx]
        pos = controller.state.entity_state.position_center[idx]

        controller.selected[entity_type].selection = [entity_idx]

        entity = getattr(controller, entity_type)[entity_idx]
        assert (jnp.equal(pos, entity.position_center).all())

        entity.visible = False

        controller.selected_entities[entity_type].y_position = 5.42
        controller.apply_changes()
        controller.update_state()

        assert controller.state.entity_state.position_center[idx][0] == pos[0]
        assert controller.state.entity_state.position_center[idx][1] == 5.42
        assert getattr(controller, entity_type)[entity_idx].position_center[1] == 5.42
