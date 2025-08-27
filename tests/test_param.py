import pytest
import jax.numpy as jnp

from vivarium.controllers.panel_controller import ParamSimulator


@pytest.mark.parametrize("scene_name, entity_type", 
                         [('braitenberg', 'agents'), 
                          ('braitenberg', 'objects'),
                          ('particle_lenia', 'particles')])
@pytest.mark.parametrize("idx", [0, 2])
def test_param_entity(scene_name, entity_type, idx, controller_and_interfaces_from_config):
    controller, interfaces = controller_and_interfaces_from_config(scene_name)

    controller_entities = controller.controllers[entity_type]
    
    entity = interfaces[entity_type].parameters  #entity_cls(controller_entities, controller.subtype_labels)
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


def test_simulator_state_param(simulator_controller_from_config):

    controller = simulator_controller_from_config('braitenberg')

    simulator_param = ParamSimulator(controller.client)
    simulator_param.update_from_server = True
    assert simulator_param.freq == controller.client.freq
    assert simulator_param.box_size == controller.client.box_size

    simulator_param.freq = -10

    assert controller.client.freq == -10
    assert controller.client.box_size == simulator_param.box_size


def test_controller_parameters(simulator_controller_from_config):
    controller = simulator_controller_from_config('braitenberg')
    controller.controllers['agents'][0].visible_wheels = False
    controller.apply_changes()
    controller.update_state()
