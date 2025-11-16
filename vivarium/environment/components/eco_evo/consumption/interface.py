import param
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData
from vivarium.environment.components.interface import Interface


class ConsumptionParam(ParameterizedData):
    source_subtype = param.String()
    target_subtype = param.String()
    range = param.Number()
    start = param.Boolean()    

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)


class ConsumptionInterface(Interface):
    def __init__(self, controller, state, panel_cls=Column):
        
        parameters = ConsumptionParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
