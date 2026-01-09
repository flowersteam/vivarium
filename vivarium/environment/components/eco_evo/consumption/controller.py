import numpy as np

from vivarium.controllers import Controller, AttributeMapping

class SingleConsumptionController:
    def __init__(self, remote, idx, global_controller_name, subtype_labels):
        self._remote = remote
        self._idx = idx
        self._global_controller_name = global_controller_name
        self._subtype_labels = subtype_labels
        
    def __getattr__(self, attr):
        if attr.startswith('_'):
            return object.__getattr__(self, attr)
        value = getattr(self._remote.state, f'{self._global_controller_name}_state').__getattr__(attr)[self._idx].item()
        if attr == 'source_subtype' or attr == 'target_subtype':
            return self._subtype_labels[value]
        return value
    
    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            if attr == 'source_subtype' or attr == 'target_subtype':
                value = np.array(self._subtype_labels.index(value), dtype=int)
            self._remote.state.__getattr__(f'{self._global_controller_name}_state').__getattr__(attr)[self._idx] = value


class ConsumptionController(Controller):
    
    def __init__(self, name, remote, mapping={}):
        self._subtype_labels = remote.controller_parameters.simulator.subtype_labels.obj()
        
        self._consumption_names = remote.controller_parameters.consumption.names.obj()
        
        self._single_consumption_controllers = {
            c_name: SingleConsumptionController(remote, idx, name, self._subtype_labels) for idx, c_name in enumerate(self._consumption_names)
        }
        
        mapping = mapping or {
            'consumption_matrix': AttributeMapping(
                f'state.{name}_state.consumption_matrix',
                remote_to_ctrl_fn=lambda x: np.array(x),
                ctrl_to_remote_fn=lambda x: np.array(x)
            )
        }        
        
        super().__init__(name, remote, mapping=mapping)

    def __getattr__(self, attr):
        if self.to_deal_with(attr):
            if attr in self._consumption_names:
                return self._single_consumption_controllers[attr]
            elif len(self._consumption_names) == 1:
                # If there is a single consumption interaction, allow direct access to its attributes
                return self._single_consumption_controllers[self._consumption_names[0]].__getattr__(attr)
            else:
                raise AttributeError(f"Access specific consumption interaction among: {list(self._consumption_names)}")
        else:            
            return super().__getattr__(attr)
    
    def __setattr__(self, attr, value):
        if self.to_deal_with(attr):
            if len(self._consumption_names) == 1:
                # If there is a single consumption interaction, allow direct access to its attributes
                if attr == 'source_subtype' or attr == 'target_subtype':
                    value = np.array(self._subtype_labels.index(value), dtype=int)
                self._remote.state.__getattr__(f'{self._name}_state').__getattr__(attr)[0] = value
            else:
                raise AttributeError(f"Access specific consumption interaction among: {list(self._consumption_names)}")
        else:
            super().__setattr__(attr, value)
            