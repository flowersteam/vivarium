from vivarium.interface.panel_app import WindowManager


def test_window_manager(simulator_from_config):
    simulator = simulator_from_config('braitenberg')
    wm = WindowManager(client=simulator, testing_mode=True)
    
    wm.interfaces['agents'].parameters.subtype = 'predator'
    wm.controller.step()
    assert wm.controller.controllers['agents'][0].subtype == 'predator'
    assert wm.controller.controllers['agents']._state.entity_state.entity_subtype[0].item() == 1
    
    wm.interfaces['agents'].parameters.behavior_0 = 'FEAR'
    wm.interfaces['agents'].parameters.sensed_prey_0 = False
    
    wm.interfaces['agents'].parameters.visible = True
    
    # assert False
