import param
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData
from vivarium.environment.components.interface import Interface


class SpawnParam(ParameterizedData):
    subtype = param.String()
    period = param.Number()
    start = param.Boolean()
    position_range = param.Tuple(default=(0, 0, 0, 0))
    orientation_range = param.Tuple(default=(0, 0))

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)


class SpawnInterface(Interface):
    def __init__(self, controller, state, panel_cls=Column):
        
        parameters = SpawnParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
