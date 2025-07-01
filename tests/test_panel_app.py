from vivarium.interface.panel_app import WindowManager
from vivarium.utils.scene_configs import SceneConfiguration


def test_window_manager():
    simulator = SceneConfiguration('braitenberg').create_simulator()
    wm = WindowManager(client=simulator, testing_mode=True)
    wm.entity_managers['agents'].selected_param_entity.subtype = 'predator'
    wm.entity_managers['agents'].selected_param_entity.behavior_0 = 'NOOP'
    wm.entity_managers['agents'].selected_param_entity.sensed_PREYS_0 = True
    # assert False
