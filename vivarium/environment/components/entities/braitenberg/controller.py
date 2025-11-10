import numpy as np

from vivarium.controllers.utils import BehaviorHandler, Logger
from vivarium.environment.components.entities.controller import EntityController
from vivarium.environment.components.entities.controller import EntityListController
from vivarium.environment.components.entities.braitenberg.behaviors import Behaviors, behavior_params


class BehaviorController:
    class Behavior:
        def __init__(self, controller, slot):
            self._controller = controller
            self._slot = slot
            
        @property
        def label(self):
            #TODO: Better make use of self._controller.controller_parameters.behaviors here
            for b in Behaviors:
                if np.equal(self._controller.behavior_params[self._slot], behavior_params[b]).all():
                    return b
            return Behaviors.CUSTOM

        @label.setter
        def label(self, behavior):
            self._controller._setitem('behavior_params', behavior_params[behavior], self._slot)

        @property
        def sensed(self):
            return [self._controller._subtype_labels[i] for i, s in enumerate(self._controller.sensed_mask[self._slot]) if bool(s)]

        @sensed.setter
        def sensed(self, subtypes):
            sensed_mask = [(s in subtypes) for s in self._controller._subtype_labels]
            self._controller._setitem('sensed_mask', np.array(sensed_mask, dtype=int), self._slot)
    
        def __next__(self):
            if self._slot < self._controller.behavior_params.shape[0] - 1:
                self._slot += 1
                return self
            else:
                raise StopIteration
    
    def __init__(self, controller):
        self._controller = controller

    def __getitem__(self, idx):
        return BehaviorController.Behavior(self._controller, idx)

    def __setitem__(self, slot, behavior):
        BehaviorController.Behavior(self._controller, slot).label = behavior
        
    def __iter__(self):
        return BehaviorController.Behavior(self._controller, 0)
            
            
class AgentController(EntityController):       

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'behavior_handler', BehaviorHandler())
        
    @property
    def behaviors(self):
        return BehaviorController(self)
    
    def stop_motors(self):
        """Stop the motors of the agent"""
        self.motor = [0, 0]
        
    def proximeters(self, sensed_entities=None):
        """Return the proximeters values of the agent"""
        #TODO: implement sensed_entities filtering (see AgentNotebookController)
        return self.prox.tolist()

    def attach_behavior(
        self, behavior_fn, name=None, interval=1, weight=1.0, start=True
    ):
        """Attach a behavior to the agent with a given weight

        :param behavior_fn: behavior_fn
        :param name: name, defaults to None
        :param interval: interval of behavior execution, defaults to 1
        :param weight: weight, defaults to 1.
        """
        self.behavior_handler.attach_behavior(
            behavior_fn, name, interval, weight, start
        )

    def detach_behavior(self, name, stop_motors=False):
        """Detach a behavior from the agent and stop the motors if needed

        :param name: name
        :param stop_motors: wether to stop the motors or not, defaults to False
        """
        self.behavior_handler.detach_behavior(name)
        if stop_motors:
            self.stop_motors()

    def detach_all_behaviors(self, stop_motors=False):
        """Detach all behaviors from the agent and stop the motors if needed

        :param stop_motors: wether to stop the motors or not, defaults to False
        """
        self.behavior_handler.detach_all_behaviors()
        if stop_motors:
            self.stop_motors()

    def start_behavior(self, name):
        """Start a behavior of the agent

        :param name: name
        """
        self.behavior_handler.start_behavior(name)

    def start_all_behaviors(self):
        """Start all behaviors of the agent"""
        self.behavior_handler.start_all_behaviors()

    def stop_behavior(self, name, stop_motors=False):
        """Stop a behavior of the agent

        :param name: name
        """
        self.behavior_handler.stop_behavior(name)
        if stop_motors:
            self.stop_motors()

    def print_behaviors(self, full_infos=False):
        """Print the behaviors and active behaviors of the agent"""
        self.behavior_handler.print_behaviors(full_infos)

    def change_behavior_weight(self, name, new_weight):
        """Change the weight of a behavior of the agent

        :param name: behavior name
        :param new_weight: new weight of the behavior
        """
        self.behavior_handler.change_behavior_weight(name, new_weight)

    def step(self, time, catch_errors):
        super().step(time, catch_errors)
        self.behave(time)

    def behave(self, time):
        """Make the agent behave according to its active behaviors

        :param time: time
        """
        self.behavior_handler.behave(self, time)
        # increment time since last meal for all alive agents

    def print_infos(self, full_infos=False):
        """Print the agent's infos

        :param full_infos: full_infos, defaults to False
        :return: agent's infos
        """
        super().print_infos()
        info_lines = []
        sensors = self.sensors()
        info_lines.append(f"Sensors: Left={sensors[0]:.2f}, Right={sensors[1]:.2f}")
        info_lines.append(
            f"Motors: Left={self.left_motor:.2f}, Right={self.right_motor:.2f}"
        )

        dict_infos = self.config.to_dict()
        if full_infos:
            info_lines.append(
                ""
            )  # add a space between other infos and eating infos atm
            info_lines.append(f"Diet: {self.diet}")
            info_lines.append(f"Eating range: {self.eating_range}")
            info_lines.append("\nConfiguration Details:")
            for k, v in dict_infos.items():
                if k not in [
                    "x_position",
                    "y_position",
                    "diameter",
                    "color",
                    "behavior",
                    "left_motor",
                    "right_motor",
                    "params",
                    "sensed",
                ]:
                    info_lines.append(f"  - {k}: {v}")

        info_lines.append("")

        return print("\n".join(info_lines))


class BraitenbergController(EntityListController):
    def __init__(self, entity_type, state, 
                 subtype_labels=None, 
                 notebook_control=False,
                 **kwargs
                 ):
        super().__init__(
            entity_type=entity_type,
            state=state,
            subtype_labels=subtype_labels,
            controller_cls=AgentController,
            notebook_control=notebook_control,
            **kwargs
        )
        
        if 'behaviors' in kwargs:
            assert len(kwargs['behaviors'][0]) <= len(self._entity_list[0].behavior_params), \
                f"Number of behaviors per agent in the config ({len(kwargs['behaviors'][0])}) exceeds the max number of behaviors in state ({len(self._entity_list[0].behavior_params)})"
            for i_agent, behaviors in enumerate(kwargs['behaviors']):
                for i_behavior, behavior in enumerate(behaviors):
                    for label, sensed in behavior.items():
                        self._entity_list[i_agent].behaviors[i_behavior].label = getattr(Behaviors, label)
                        self._entity_list[i_agent].behaviors[i_behavior].sensed = sensed
