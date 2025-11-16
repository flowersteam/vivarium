import param
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData
from vivarium.environment.components.interface import Interface


class ConsumptionParam(ParameterizedData):
    source_subtype = param.Selector()
    target_subtype = param.Selector()
    range = param.Number()
    start = param.Boolean()    

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)
        self.param.source_subtype.objects = controller._subtype_labels
        self.param.target_subtype.objects = controller._subtype_labels


class ConsumptionInterface(Interface):
    def __init__(self, controller, panel_cls=Column):
        
        parameters = ConsumptionParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
