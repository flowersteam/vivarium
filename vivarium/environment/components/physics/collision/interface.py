import param
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData
from vivarium.environment.components.interface import Interface


class CollisionParam(ParameterizedData):
    epsilon = param.Number()
    alpha = param.Number()

    def __init__(self, controller, **params):
        super().__init__(data=controller, **params)


class CollisionInterface(Interface):
    def __init__(self, controller, state, panel_cls=Column):
        
        parameters = CollisionParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
