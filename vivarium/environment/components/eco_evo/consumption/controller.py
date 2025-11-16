import numpy as np

from vivarium.controllers import Controller, AttributeMapping, set_nested_attr

class ConsumptionController(Controller):
    
    def __init__(self, name, remote, mapping={}):
        self._subtype_labels = remote.controller_parameters.simulator.subtype_labels.obj()
        
        mapping = mapping or {
            'source_subtype': AttributeMapping(
                f'state.{name}_state.source_subtype',
                remote_to_ctrl_fn=lambda x: self._subtype_labels[x],
                ctrl_to_remote_fn=lambda x: np.array(self._subtype_labels.index(x), dtype=int)
            ),
            'target_subtype': AttributeMapping(
                f'state.{name}_state.target_subtype',
                remote_to_ctrl_fn=lambda x: self._subtype_labels[x],
                ctrl_to_remote_fn=lambda x: np.array(self._subtype_labels.index(x), dtype=int)
            ),
        }        
        super().__init__(name, remote, mapping=mapping)

    def __getattr__(self, attr):
        if self.to_deal_with(attr):
            return getattr(self._remote.state, f'{self._name}_state').__getattr__(attr)
        else:            
            return super().__getattr__(attr)
    
    def __setattr__(self, attr, value):
        if self.to_deal_with(attr):
            attr = f'state.{self._name}_state.{attr}'
            set_nested_attr(self._remote, attr, value)
        else:
            super().__setattr__(attr, value)
            