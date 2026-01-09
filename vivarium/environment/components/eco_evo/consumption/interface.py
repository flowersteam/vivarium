import param
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData
from vivarium.environment.components.interface import Interface


class SingleConsumptionParam(ParameterizedData):
    source_subtype = param.Selector()
    target_subtype = param.Selector()
    range = param.Number()
    start = param.Boolean()    

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)
        self.param.source_subtype.objects = controller._subtype_labels
        self.param.target_subtype.objects = controller._subtype_labels

class ConsumptionParam(ParameterizedData):
    
    def __init__(self, controller):
        super().__init__(controller=controller)
        self._single_consumption_params = [
            SingleConsumptionParam(controller=single_controller, name=name)
            for name, single_controller in controller._single_consumption_controllers.items()
        ]
        self.param.add_parameter(
            'consumption',
            param.Selector(
                objects=self._single_consumption_params,
                default=self._single_consumption_params[0]
            )
        )


class ConsumptionInterface(Interface):
    def __init__(self, controller, panel_cls=Column):
        
        parameters = ConsumptionParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
