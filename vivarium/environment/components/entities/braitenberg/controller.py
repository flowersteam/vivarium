import numpy as np

from vivarium.controllers.utils import BehaviorHandler, Logger
from vivarium.environment.components.entities.controller import EntityController
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environment.components.entities.braitenberg.interface import ParamAgent
from vivarium.environment.components.entities.braitenberg.interface import AgentManager
from vivarium.controllers.notebook_controller import NotebookControllerEntity
from vivarium.environment.components.entities.braitenberg.behaviors import Behaviors, behavior_to_params


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


class Agent(NotebookControllerEntity):
    """Agent class that represents an agent in the simulation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'behavior_handler', BehaviorHandler())
        object.__setattr__(self, 'logger', Logger())
        object.__setattr__(self, 'eating_range', 10)
        object.__setattr__(self, 'diet', [])
        object.__setattr__(self, 'ate', False)
        object.__setattr__(self, 'time_since_feeding', np.inf)
        object.__setattr__(self, 'simulation_entities', None)
        self.set_manual()

    def set_manual(self):
        """Set the agent's behavior to manual"""
        self.behavior = np.full(
            shape=self.behavior.shape, fill_value=Behaviors.MANUAL.value
        )
        self.stop_motors()

    def sensors(self, sensed_entities=None):
        """Return the sensors values of the agent

        :param sensed_entities: sensed_entities of the sensors under the form of strings, defaults to None
        :return: sensors values
        """
        left, right = self.prox
        if sensed_entities is not None:
            # TODO: transform the strings of sensed entities into ints (this fn can surely be optimized)
            assert all(
                ent_subtype in self.valid_subtypes for ent_subtype in sensed_entities
            ), f"Please specify valid sensed entities among {self.valid_subtypes}"
            sensed_entities = [
                self._subtype_label_to_idx[label] for label in sensed_entities
            ]
            sensed_type_left, sensed_type_right = self.prox_sensed_ent_type
            left = left if sensed_type_left in sensed_entities else 0
            right = right if sensed_type_right in sensed_entities else 0
        return [left, right]

    def entity_sensors(self):
        """Return the left and right sensed entities of the agent if they are sensed, else None

        :return: sensed entities
        """
        left_idx, right_idx = self.prox_sensed_ent_idx
        left_ent = (
            self.simulation_entities[left_idx] if self.config.left_prox != 0 else None
        )
        right_ent = (
            self.simulation_entities[right_idx] if self.config.right_prox != 0 else None
        )
        return [left_ent, right_ent]

    def attribute_sensors(self, sensed_attribute, default_value=None):
        """Return the sensed attribute of the left and right sensed entities

        :param sensed_attribute: sensed_attribute
        :param default_value: default value if the attribute is not found, defaults to None
        :return: sensed attributes
        """
        left_ent, right_ent = self.entity_sensors()
        # get the sensed attribute of the entities with a getattr, specify a default value if the attribute is not found
        return (
            getattr(left_ent, sensed_attribute, default_value),
            getattr(right_ent, sensed_attribute, default_value),
        )

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
        object.__setattr__(self, 'time_since_feeding', self.time_since_feeding + 1)

    def stop_motors(self):
        """Stop the motors of the agent"""
        self.motor = [0, 0]

    def has_eaten(self):
        """Check if the agent has eaten

        :return: True if the agent has eaten, False otherwise
        """
        val = self.ate
        self.ate = False
        return val

    # TODO : maybe delete this function
    def has_eaten_since(self, time):
        """Check if the agent has eaten since a given time

        :return: True if the agent has eaten since the given time, False otherwise
        """
        return self.time_since_feeding <= time

    def add_log(self, log_field, data):
        """Add a log to the agent's logger (e.g robot.add_log("left_prox", left_prox_value))

        :param log_field: log_field of the log
        :param data: data logged
        """
        self.logger.add(log_field, data)

    def get_log(self, log_field):
        """Get the log of the agent's logger for a specific log_field

        :param log_field: desired log_field
        :return: associated log_field data
        """
        return self.logger.get_log(log_field)

    def clear_all_logs(self):
        """Clear all logs of the agent's logger

        :return: cleared logs
        """
        return self.logger.clear()

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


class BraitenbergController(EntityController):
    def __init__(self, entity_type, 
                 controller_cls=ControllerAgent, 
                 notebook_controller_cls=Agent,
                 param_cls=ParamAgent,
                 render_cls=AgentManager,
                 **kwargs):
        super().__init__(
            entity_type=entity_type,
            controller_cls=controller_cls,
            notebook_controller_cls=notebook_controller_cls,
            param_cls=param_cls,
            render_cls=render_cls,
            **kwargs
        )
