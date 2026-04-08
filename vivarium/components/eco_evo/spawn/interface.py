import param
from panel.layout import Column

from vivarium.interface.parameterized import ParameterizedData
from vivarium.components.interface import Interface


class SingleSpawnParam(ParameterizedData):
    subtype = param.Selector()
    period = param.Number()
    start = param.Boolean()
    position_range = param.Tuple(default=(0, 0, 0, 0))
    orientation_range = param.Tuple(default=(0, 0))

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)
        self.param.subtype.objects = controller._subtype_labels


class SpawnParam(ParameterizedData):

    def __init__(self, controller):
        super().__init__(controller=controller)
        self._single_spawn_params = [
            SingleSpawnParam(controller=single_controller, name=name)
            for name, single_controller in controller._single_spawn_controllers.items()
        ]
        self.param.add_parameter(
            'spawn',
            param.Selector(
                objects=self._single_spawn_params,
                default=self._single_spawn_params[0]
            )
        )


class SpawnInterface(Interface):
    def __init__(self, controller, panel_cls=Column):

        parameters = SpawnParam(controller=controller)

        super().__init__(controller, parameters, panel_cls=panel_cls)
