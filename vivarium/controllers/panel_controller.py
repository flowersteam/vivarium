import param
import logging

import jax.numpy as jnp

from vivarium.controllers.simulator_controller import (
    SimulatorController, ControllerObject
)
from vivarium.controllers.dataclass_wrapper import SimulatorParametersWrapper


lg = logging.getLogger(__name__)


class PanelSimulatorParametersWrapper(SimulatorParametersWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'hide_non_existing', True)
        object.__setattr__(self, 'config_update', False)
        object.__setattr__(self, 'panel_parameters', ['hide_non_existing', 'config_update'])

    def __getattr__(self, attr):
        if attr in self.panel_parameters:
            return object.__getattr__(self, attr)
        return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if attr in self.panel_parameters:
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)

# TODO: What's the purpose of this?
class PanelControllerObject(ControllerObject):
    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)


class ParameterizedData(param.Parameterized):
    update_from_server = param.Event()

    def __init__(self, data, parameter_mapping={}, panel_parameters=[], **params):
        super().__init__(**params)
        self.data = data
        self.selection = None
        self.parameter_mapping = parameter_mapping
        self.panel_parameters = panel_parameters
        self.update_parameter_list()
        self.panel_visibility_parameters = [p for p in self.panel_parameters if p.startswith('visible')]
        self.param_to_jax = {p: self.parameter_mapping[p] if p in self.parameter_mapping else ParameterMapping(p) for p in self.parameters}
        self.jax_to_param = {p.jax_name: p for p in self.param_to_jax.values()}
        self.param.watch(self.update_to, self.parameters, onlychanged=True)
        self.param.watch(self.udpate_panel_parameter, self.panel_parameters, onlychanged=True)

    @param.depends('update_from_server', watch=True)
    def update_from(self):
        self.allow_update_to = False  # Prevents to call update_to callback for each updated parameter
        data = self.data if self.selection is None else self.data[self.selection[0]]
        for p, mapping in self.param_to_jax.items():
            setattr(self, p, mapping.jax_to_param_fn(getattr(data, mapping.jax_name)))
        self.allow_update_to = True

    def update_to(self, event):
        if self.allow_update_to:
            if event.name in self.param_to_jax:
                mapping = self.param_to_jax[event.name]
                if self.selection is None:
                    setattr(self.data,
                            mapping.param_name, mapping.param_to_jax_fn(event.new))
                    return
                for idx in self.selection:
                    setattr(self.data[idx],
                            mapping.param_name, mapping.param_to_jax_fn(event.new))

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


class ParameterMapping:
    def __init__(self, jax_name, param_name=None, jax_to_param_fn=None, param_to_jax_fn=None):
        self.jax_name = jax_name
        self.param_name = param_name if param_name is not None else jax_name
        self.jax_to_param_fn = jax_to_param_fn if jax_to_param_fn is not None else lambda x: x
        self.param_to_jax_fn = param_to_jax_fn if param_to_jax_fn is not None else lambda x: x


class ParamSimulator(ParameterizedData):
    box_size = param.Number()
    num_scan_steps = param.Integer()
    freq = param.Number()
    neighbor_radius = param.Number()
    to_jit = param.Boolean()
    hide_non_existing = param.Boolean(True)
    config_update = param.Boolean(False)

    def __init__(self, simulator_state_wrapper, **params):
        
        parameter_mapping = {}
        super().__init__(simulator_state_wrapper, 
                         parameter_mapping=parameter_mapping, 
                         panel_parameters=['hide_non_existing', 'config_update'],
                         **params)


entity_parameter_mapping = {
    'orientation': ParameterMapping('position_orientation'),
    'mass': ParameterMapping(
        'mass_center',
        jax_to_param_fn=lambda x: x[0].item(),
        param_to_jax_fn=lambda x: jnp.array([x])
    ),
    'exists': ParameterMapping(
        'exists',
        jax_to_param_fn=lambda x: bool(x.item()),
        param_to_jax_fn=lambda x: jnp.array(int(x))
    ),
}

class ParamEntity(ParameterizedData):
    x_position = param.Number()
    y_position = param.Number()
    orientation = param.Number()
    mass = param.Number()
    diameter = param.Number()
    friction = param.Number()
    exists = param.Boolean()
    color = param.Color()
    visible = param.Boolean()

    def __init__(self, entities, subtype_labels, panel_parameters=[], **params):
        super().__init__(entities, 
                         parameter_mapping=entity_parameter_mapping,
                         panel_parameters=panel_parameters + ['visible', 'color'],
                         **params)
        self.selection = [0]
        #Note: subtype labels are not used yet but should (to set in the interface the subtype of entities)
        # But should they be part of panel_parameters?

    @property
    def selected_entity_data(self):
        return self.data[self.selection[0]]

def behavior_param_name(b_idx):
    return f'behavior_{b_idx}'

def sensed_param_name(label, b_idx):
    return f'sensed_{label}_{b_idx}'

class Selected(param.Parameterized):
    """Class to store the selected entities in the interface"""

    selection = param.ListSelector([0], objects=[0])

    def __len__(self):
        return len(self.selection)


class PanelController(SimulatorController):
    """Controller for the panel interface"""
    # config_field = 'panel_controller'
    def __init__(self, client=None, subtypes=[], **controllers):
        
        super().__init__(client=client, subtypes=subtypes, **controllers)

        self.selected = {etype: Selected() for etype in controllers.keys()}
        
        self.selected_entities = {etype: controller.param_cls(self.entity_lists[etype], self.subtype_labels) 
                                  for etype, controller in controllers.items()}

        self.param_simulator = ParamSimulator(self.simulator_parameters)
        
        for s_ent in self.selected_entities.values():
            s_ent.update_from_server = True
        self.param_simulator.update_from_server = True

        self.update_selected()
        for selected in self.selected.values():
            selected.param.watch(
                self.pull_selected_entities,
                ["selection"],
                onlychanged=True,
                precedence=1,
            )

    def create_simulator_parameters_wrapper(self):
        self.simulator_parameters = PanelSimulatorParametersWrapper(self.simulator_parameters)

    def update_selected(self, *events):
        """Update the entity list"""
        state = self.state
        for etype, selected in self.selected.items():
            selected.param.selection.objects = state.entity_type_idx(etype).tolist()

    def pull_selected_entities(self, *events):
        """Pull the selected configurations"""
        for etype, selected in self.selected.items():
            self.selected_entities[etype].selection = selected.selection
            self.selected_entities[etype].update_from_server = True

    def pull_all_data(self):
        """Pull all the data from the simulator"""
        self.update_state()
        self.update_entity_lists()
        self.pull_selected_entities()
