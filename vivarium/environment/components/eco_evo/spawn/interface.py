import param
from panel.layout import Column

from vivarium.interface.parameterized import ParameterizedData
from vivarium.environment.components.interface import Interface


class SpawnParam(ParameterizedData):
    subtype = param.Selector()
    period = param.Number()
    start = param.Boolean()
    position_range = param.Tuple(default=(0, 0, 0, 0))
    orientation_range = param.Tuple(default=(0, 0))

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)
        self.param.subtype.objects = controller._subtype_labels


class SpawnInterface(Interface):
    def __init__(self, controller, panel_cls=Column):
        
        parameters = SpawnParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
