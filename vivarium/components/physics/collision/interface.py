import param
from panel.layout import Column

from vivarium.interface.parameterized import ParameterizedData
from vivarium.components.interface import Interface


class CollisionParam(ParameterizedData):
    epsilon = param.Number()
    alpha = param.Number()

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)


class CollisionInterface(Interface):
    def __init__(self, controller, panel_cls=Column):
        
        parameters = CollisionParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
