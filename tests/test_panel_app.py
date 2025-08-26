from vivarium.interface.panel_app import WindowManager


def test_window_manager(simulator_from_config):
    simulator = simulator_from_config('braitenberg')
    wm = WindowManager(client=simulator, testing_mode=True)
    wm.interfaces['agents'].parameters.subtype = 'predator'
    wm.interfaces['agents'].parameters.behavior_0 = 'NOOP'
    wm.interfaces['agents'].parameters.sensed_PREYS_0 = True
    # assert False
