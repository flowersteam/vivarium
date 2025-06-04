from vivarium.controllers.panel_controller import ParamEntity
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environments.entities.controller import EntityController
from vivarium.environments.entities.objects.interface import ObjectManager

class ObjectController(EntityController):
    def __init__(self, entity_type, 
                 controller_cls=ControllerEntity, 
                 param_cls=ParamEntity,
                 render_cls=ObjectManager,
                 **kwargs):
        super().__init__(
            entity_type=entity_type,
            controller_cls=controller_cls,
            param_cls=param_cls,
            render_cls=render_cls,
            **kwargs
        )
