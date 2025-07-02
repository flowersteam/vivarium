import numpy as np

from vivarium.environments.entities.controller import EntityController
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.controllers.notebook_controller import Agent
from vivarium.environments.entities.braitenberg.interface import ParamAgent
from vivarium.environments.entities.braitenberg.interface import AgentManager
from vivarium.environments.entities.braitenberg.behaviors import Behaviors, behavior_to_params


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


class BraitenbergController(EntityController):
    def __init__(self, entity_type, 
                 controller_cls=ControllerAgent, 
                 param_cls=ParamAgent,
                 render_cls=AgentManager,
                 notebook_controller_cls=Agent,
                 **kwargs):
        super().__init__(
            entity_type=entity_type,
            controller_cls=controller_cls,
            param_cls=param_cls,
            render_cls=render_cls,
            **kwargs
        )
        self.notebook_controller_cls = notebook_controller_cls
