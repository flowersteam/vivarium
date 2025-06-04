from vivarium.controllers.dataclass_wrapper import EntityList, create_dataclass_from_dict
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environments.entities.braitenberg.behaviors import Behaviors, behavior_to_params


import numpy as np

from vivarium.environments.entities.braitenberg.interface import ParamAgent
from vivarium.environments.entities.braitenberg.interface import AgentManager


class ControllerAgent(ControllerEntity):

    def set_behavior(self, slot_idx, behavior, sensed):
        assert slot_idx < self.behavior.shape[0], 'Behavior index out of bounds'
        if isinstance(behavior, Behaviors):
            behavior = behavior.value
        cur_behaviors = np.array(self.behavior)
        cur_behaviors[slot_idx] = behavior
        self.behavior = cur_behaviors
        cur_sensed = np.array(self.sensed)
        cur_sensed[slot_idx] = [int(i in sensed) for i in range(len(cur_sensed[slot_idx]))]
        self.sensed = cur_sensed
        cur_params = np.array(self.behavior_params)
        cur_params[slot_idx] = behavior_to_params(behavior)
        self.behavior_params = cur_params


# TODO: What's the purpose of this class? Not uses at the moment (May, 31, 2025) but the whole pipeline seems to work anyway.
class PanelControllerAgent(ControllerAgent):
    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return object.__getattr__(self, attr)
        return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if attr in self.__dict__:
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)


class BraitenbergController:
    def __init__(self, entity_type, **kwargs):
        self.entity_type = entity_type
        self.controller_parameters = create_dataclass_from_dict(
            'ControllerParameters',
            kwargs)
        self.param_cls = ParamAgent
        self.render_cls = AgentManager

    def controller(self, state):  #, state):
        etype_int = getattr(state, self.entity_type).entity_type
        return EntityList(
            state=state, entity_type=self.entity_type, entity_type_idx=etype_int,
            entity_wrapper_list=[
                ControllerAgent(state, idx, self.entity_type,
                                controller_parameters=self.controller_parameters[int(state.entity_state.entity_type_idx[idx])])
                    #    **{attr: val[int(state.entity_state.entity_type_idx[idx])]
                    #       for attr, val in etype_to_kwargs[etype].items()}
                    #    ) 
                for idx, type in enumerate(state.entity_state.entity_type)
                if type == etype_int]
        )
