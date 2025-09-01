import param
import logging

from vivarium.controllers.dataclass_wrapper import ChangeRecorder, update_dataclass


# TODO: (2025-08-26) Rename this file, which will only contain code for param<->simulator/state communication


lg = logging.getLogger(__name__)


class ParameterizedData(param.Parameterized):
    update_from_server = param.Event()

    def __init__(self, data, panel_parameters=[], **params):
        super().__init__(**params)
        self.data = data
        self.selection = None
        self.panel_parameters = panel_parameters
        self.update_parameter_list()
        
        self.param.watch(self.update_to, self.parameters, onlychanged=True)
        self.param.watch(self.udpate_panel_parameter, self.panel_parameters, onlychanged=True)

    @param.depends('update_from_server', watch=True)
    def update_from(self):
        self.allow_update_to = False  # Prevents to call update_to callback for each updated parameter
        data = self.data if self.selection is None else self.data[self.selection[0]]
        for p in self.parameters:
            setattr(self, p, getattr(data, p))
        self.allow_update_to = True

    def update_to(self, event):
        if self.allow_update_to:
            if self.selection is None:
                setattr(self.data,
                        event.name, event.new)
                return
            for idx in self.selection:
                setattr(self.data[idx],
                        event.name, event.new)

    def udpate_panel_parameter(self, event):
        if self.selection is None:
            setattr(self.data, event.name, event.new)
            return
        for idx in self.selection:
            setattr(self.data[idx], event.name, event.new)

    def update_parameter_list(self):
        parameters = self.to_dict(exclude=['name', 'update_from_server'] + self.panel_parameters)
        self.parameters =list(parameters.keys())

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


class ParamSimulator(ParameterizedData):
    box_size = param.Number()
    num_scan_steps = param.Integer()
    freq = param.Number()
    neighbor_radius = param.Number()
    to_jit = param.Boolean()
    hide_non_existing = param.Boolean(True)
    config_update = param.Boolean(False)

    def __init__(self, simulator_state_wrapper, **params):
        
        super().__init__(simulator_state_wrapper, 
                         panel_parameters=['hide_non_existing', 'config_update'],
                         **params)


class SimulatorParametersWrapper:
    def __init__(self, simulator_parameters):
        object.__setattr__(self, '_simulator_parameters', simulator_parameters)
        object.__setattr__(self, '_change_recorder', ChangeRecorder())

    def __getattr__(self, attr):
        return getattr(self._simulator_parameters, attr)

    def _setitem(self, attr, value, idx=None):
        setattr(self._change_recorder, attr, value)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)

    def fetch_changes(self):
        changes = self._change_recorder.fetch_changes()
        return changes

    def apply_to_state(self, simulator):  # TODO: change method name
        changes = self.fetch_changes()
        self._simulator = update_dataclass(simulator, changes)
        self._change_recorder = ChangeRecorder()
        return self._simulator



class PanelSimulatorParametersWrapper(SimulatorParametersWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'config_update', False)
        object.__setattr__(self, 'panel_parameters', ['config_update'])

    def __getattr__(self, attr):
        if attr in self.panel_parameters:
            return object.__getattr__(self, attr)
        return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if attr in self.panel_parameters:
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)
            
       
# class PanelController_: #(SimulatorController):
#     """Controller for the panel interface"""
#     def __init__(self, client=None, subtypes=[], **controllers):
        
#         super().__init__(client=client, subtypes=subtypes, **controllers)

#         # self.selected = {etype: Selected() for etype in controllers.keys()}
        
#         # self.selected_entities = {etype: controller.param_cls(self.entity_lists[etype], self.subtype_labels) 
#         #                           for etype, controller in controllers.items()}

#         # self.param_simulator = ParamSimulator(self.simulator_parameters)
        
#         # for s_ent in self.selected_entities.values():
#         #     s_ent.update_from_server = True
#         # self.param_simulator.update_from_server = True

#         # self.update_selected()
#         # for selected in self.selected.values():
#         #     selected.param.watch(
#         #         self.pull_selected_entities,
#         #         ["selection"],
#         #         onlychanged=True,
#         #         precedence=1,
#         #     )
    
#     def create_simulator_parameters_wrapper(self):
#         self.simulator_parameters = PanelSimulatorParametersWrapper(self.simulator_parameters)

#     # def update_selected(self, *events):
#     #     """Update the entity list"""
#     #     state = self.state
#     #     for etype, selected in self.selected.items():
#     #         selected.param.selection.objects = state.entity_type_idx(etype).tolist()

#     # def pull_selected_entities(self, *events):
#     #     """Pull the selected configurations"""
#     #     for etype, selected in self.selected.items():
#     #         self.selected_entities[etype].selection = selected.selection
#     #         self.selected_entities[etype].update_from_server = True

#     def pull_all_data(self):  # TODO: No longer needed?
#         """Pull all the data from the simulator"""
#         self.update_state()
#         self.update_entity_lists()
#         self.pull_selected_entities()
