import pytest
from vivarium.interface.panel_app import WindowManager

@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_window_manager(client_fixture, vivarium_controller, request):
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = vivarium_controller(client)
    wm = WindowManager(controller=controller, testing_mode=True)
    
    wm.interfaces['agents'].parameters.subtype = 'predator'
    wm.controller.simulator_step()
    assert wm.controller.controllers['agents'][0].subtype == 'predator'
    assert wm.controller.client.state.entity_state.entity_subtype[0].item() == 1
    
    wm.interfaces['agents'].parameters.behavior_0 = 'FEAR'
    wm.interfaces['agents'].parameters.sensed_prey_0 = False
    
    wm.interfaces['agents'].parameters.visible = True
    
    wm.update_plot_cb()
    
    # assert False
