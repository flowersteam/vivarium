import param
import logging
from dataclasses import asdict


lg = logging.getLogger(__name__)


class ParameterizedData(param.Parameterized):
    update_from_server = param.Event()

    def __init__(self, controller, **params):
        super().__init__(**params)
        self.controller = controller
        self.selection = None
        self.direct_mapping_parameters = self._direct_mapping_parameters()
        self.param.watch(self.update_to, self.direct_mapping_parameters, onlychanged=True)
        self.allow_update_to = True

    @param.depends('update_from_server', watch=True)
    def update_from(self):
        self.allow_update_to = False  # Prevents to call update_to callback for each updated parameter
        data = self.controller if self.selection is None else self.controller[self.selection[0] if len(self.selection) else 0]
        for p in self.direct_mapping_parameters:
            if isinstance(getattr(self, p), ParameterizedData):
                getattr(self, p).update_from()
            else:
                if not getattr(self.param, p).constant:
                    setattr(self, p, getattr(data, p))
        self.allow_update_to = True

    def update_to(self, event):
        if self.allow_update_to:
            if self.selection is None:
                setattr(self.controller,
                        event.name, event.new)
                return
            for idx in self.selection:
                setattr(self.controller[idx],
                        event.name, event.new)


    def _direct_mapping_parameters(self):
        parameters = self.to_dict(exclude=['name', 'update_from_server'])
        return list(parameters.keys())

    def to_dict(self, params=None, exclude=['name']):
        """Return a dictionary with the configuration parameters

        :param params: params, defaults to None
        :return: dictionary with the configuration parameters
        """
        d = self.param.values()
        for e in exclude:
            del d[e]
        if params is not None:
            return {p: d[p] for p in params}
        else:
            return d

    def param_names(self, exclude=['name']):
        """Return the names of the configuration parameters

        :return: list of parameter names
        """
        return list(self.to_dict(exclude=exclude).keys())

    def json(self):
        """Return a JSON representation of the configuration

        :return: JSON representation of the configuration
        """
        return self.param.serialize_parameters(subset=self.param_names())

class ParamEnvironment(ParameterizedData):
    box_size = param.Number()
    num_scan_steps = param.Integer()
    neighbor_radius = param.Number()
    to_jit = param.Boolean()

    def __init__(self, environment_controller):
        
        super().__init__(environment_controller, 
                         **asdict(environment_controller._obj))

class ParamSimulator(ParameterizedData):
    freq = param.Number()
    scene_name = param.String()
    simulation_running = param.Boolean()
    run_from = param.Selector()
    subtype_labels = param.List(constant=True)

    # config_update = param.Boolean(False)

    env = param.ClassSelector(class_=ParamEnvironment)


    def __init__(self, simulator_controller):
        
        params = asdict(simulator_controller._remote.controller_parameters.simulator.obj())
        params.pop('client_names', None)
        params.pop('close', None)
        params['env'] = ParamEnvironment(simulator_controller.env)
        
        super().__init__(simulator_controller, 
                         **params)
        
        self.param.run_from.objects = simulator_controller.client_names + ['server']
        self.param.run_from.default = params['run_from']
