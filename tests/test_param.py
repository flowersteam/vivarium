import pytest
import jax.numpy as jnp

from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.controllers.panel_controller import ParamSimulator
from vivarium.controllers.simulator_controller import SimulatorController


@pytest.mark.parametrize("scene_name, entity_type", 
                         [('braitenberg', 'agents'), 
                          ('braitenberg', 'objects'),
                          ('particle_lenia', 'particles')])
@pytest.mark.parametrize('rigid_body', [False, True])
@pytest.mark.parametrize("idx", [0, 2])
def test_param_entity(scene_name, entity_type, rigid_body, idx):
    scene_config = SceneConfiguration(scene_name)
    state = scene_config.create_state(rigid_body=rigid_body)
    simulator = scene_config.create_simulator(state=state)
    controller = SimulatorController(simulator)

    controller_entities = getattr(controller, entity_type)
    
    entity_cls = scene_config.entity_type_client_configs[entity_type].param.cls
    entity = entity_cls(controller_entities, controller.subtype_labels)
    entity.selection = [idx]
    entity.update_from_server = True

    assert entity.x_position == controller_entities[idx].position_center[0]
    entity.x_position = 10
    assert entity.x_position == 10
    controller.apply_changes()
    controller.update_state()
    assert controller_entities[idx].position_center[0] == 10
    entity.exists = False
    controller.apply_changes()
    controller.update_state()
    assert controller_entities[idx].exists.item() == 0

    if entity_type == 'agents':
        entity.right_motor = 2.
        controller.apply_changes()
        controller.update_state()
        assert controller_entities[idx].motor[1] == 2.

    controller_entities[idx].position_orientation = 1.
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.orientation == 1.

    controller_entities[idx].exists = jnp.array(1)
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.exists

    controller_entities[idx].x_position = 20
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.x_position == 20

    if entity_type == 'agents':
        controller_entities[idx].left_motor = 4.
        controller.apply_changes()
        controller.update_state()
        entity.update_from_server = True
        assert entity.left_motor == 4.


def test_simulator_state_param():
    simulator = SceneConfiguration('braitenberg').create_simulator()
    controller = SimulatorController(simulator)

    simulator_param = ParamSimulator(controller.client)
    simulator_param.update_from_server = True
    assert simulator_param.freq == controller.client.freq
    assert simulator_param.box_size == controller.client.box_size

    simulator_param.freq = -10

    assert controller.client.freq == -10
    assert controller.client.box_size == simulator_param.box_size
