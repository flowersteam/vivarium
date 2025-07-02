from vivarium.environments.entities.controller import EntityController
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.controllers.notebook_controller import Object


class ControllerObject(ControllerEntity):
    pass
# TODO: What's the purpose of this class? Not uses at the moment (May, 31, 2025) but the whole pipeline seems to work anyway.
# (Idem in braintenberg component.py)
class PanelControllerObject(ControllerObject):
    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)


class ObjectController(EntityController):
    def __init__(self, entity_type, 
                 notebook_controller_cls=Object,
                 **kwargs):
        super().__init__(
            entity_type=entity_type,
            **kwargs
        )
        self.notebook_controller_cls = notebook_controller_cls
