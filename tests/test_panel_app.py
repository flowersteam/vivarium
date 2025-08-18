from vivarium.interface.panel_app import WindowManager


def test_window_manager(simulator_from_config):
    simulator = simulator_from_config('braitenberg')
    wm = WindowManager(client=simulator, testing_mode=True)
    wm.entity_managers['agents'].selected_param_entity.subtype = 'predator'
    wm.entity_managers['agents'].selected_param_entity.behavior_0 = 'NOOP'
    wm.entity_managers['agents'].selected_param_entity.sensed_PREYS_0 = True
    # assert False
