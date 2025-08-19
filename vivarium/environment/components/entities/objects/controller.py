from vivarium.environment.components.entities.controller import EntityController
from vivarium.controllers.simulator_controller import ControllerEntity


# This module is not used at the moment (June 3, 2025).

class ControllerObject(ControllerEntity):
    pass


# TODO: What's the purpose of this class? Not uses at the moment (May, 31, 2025) but the whole pipeline seems to work anyway.
# (Idem in braintenberg component.py)
class PanelControllerObject(ControllerObject):
    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)


