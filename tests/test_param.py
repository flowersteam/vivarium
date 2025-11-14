import pytest
import jax.numpy as jnp

from vivarium.controllers.panel_controller import ParamSimulator


@pytest.mark.parametrize("client_fixture, scene_name, entity_type", 
                         [('grpc_client', 'braitenberg', 'agents'), 
                          ('simulator_from_config', 'braitenberg', 'objects'),
                          ('simulator_from_config', 'particle_lenia', 'particles')])
@pytest.mark.parametrize("idx", [0, 2])
def test_param_entity(client_fixture, scene_name, entity_type, idx, controller_and_interfaces_from_config, request):
    client = request.getfixturevalue(client_fixture)(scene_name)
    controller, interfaces = controller_and_interfaces_from_config(client)

    controller_entities = controller.controllers[entity_type]
    
    entity = interfaces[entity_type].parameters
    entity.selection = [idx]
    entity.update_from_server = True

    assert entity.x_position == controller_entities[idx].position_center[0]
    entity.x_position = 10
    assert entity.x_position == 10
    controller.apply_changes()
    assert controller_entities[idx].position_center[0] == 10
    entity.exists = False
    controller.apply_changes()
    assert not controller_entities[idx].exists

    if entity_type == 'agents':
        entity.right_motor = 2.
        controller.apply_changes()
        assert controller_entities[idx].motor[1] == 2.

    controller_entities[idx].position_orientation = 1.
    controller.apply_changes()
    entity.update_from_server = True
    assert entity.orientation == 1.

    controller_entities[idx].exists = True
    controller.apply_changes()
    entity.update_from_server = True
    assert entity.exists

    controller_entities[idx].x_position = 20
    controller.apply_changes()
    entity.update_from_server = True
    assert entity.x_position == 20

    if entity_type == 'agents':
        controller_entities[idx].left_motor = 4.
        controller.apply_changes()
        entity.update_from_server = True
        assert entity.left_motor == 4.


def test_controller_param(controller_and_interfaces_from_config, simulator_from_config):
    controller, interfaces = controller_and_interfaces_from_config(simulator_from_config('braitenberg'))

    param = interfaces['collision'].parameters

    param.update_from_server = True
    assert param.epsilon == controller.client.state.collision_state.epsilon.item()
    assert param.alpha == controller.client.state.collision_state.alpha.item()

    param.epsilon = 42.
    param.alpha = 43.

    controller.apply_changes()
    assert controller.client.state.collision_state.epsilon.item() == 42.
    assert controller.client.state.collision_state.alpha.item() == 43.


def test_simulator_state_param(simulator_controller, simulator_from_config):

    controller = simulator_controller(simulator_from_config('braitenberg'))

    simulator_param = ParamSimulator(controller.simulator)
    simulator_param.update_from_server = True
    assert simulator_param.freq == controller.simulator.freq
    assert simulator_param.env.box_size == controller.simulator.env.box_size

    simulator_param.freq = -10
    
    controller.apply_changes()

    assert controller.simulator.freq == -10
    assert controller.simulator.env.box_size == simulator_param.env.box_size
    
    
    simulator_param.env.box_size = 41.
    
    controller.apply_changes()
    
    assert controller.simulator.env.box_size == 41.
    assert controller.simulator.freq == -10


def test_controller_parameters(simulator_controller, simulator_from_config):
    controller = simulator_controller(simulator_from_config('braitenberg'))
    controller.controllers['agents'][0].visible_wheels = False
    controller.apply_changes()
