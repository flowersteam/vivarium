import time
import math
import threading
import functools
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

import numpy as np
import logging

from vivarium.environments.braitenberg.behaviors import Behaviors
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.simulator_states import StateType, EntityType
from vivarium.controllers.utils import Logger, RoutineHandler, BehaviorHandler

lg = logging.getLogger(__name__)

if logging.root.handlers:
    lg.setLevel(logging.root.level)
else:
    lg.setLevel(logging.WARNING)


class Entity:
    """Entity class that represents an entity in the simulation"""

    def __init__(self, config):
        self.config = config
        self.routine_handler = RoutineHandler()
        self.user_events = {}

    def __getattr__(self, item):
        """Get the attribute of the entity

        :param item: item
        :return: attribute
        """
        if item in self.config.param_names():
            return getattr(self.config, item)
        else:
            return super().__getattr__(item)

    # TODO : Add a check to ensure that the attribute's value is authorized (according to params bounds)
    def __setattr__(self, item, val):
        """Set the attribute of the entity

        :param item: field to set
        :param val: value to set
        :return: attribute
        """
        if item != "config" and item in self.config.param_names():
            self.user_events[item] = val  # ensures the event is set during the run loop
        else:
            return super().__setattr__(item, val)

    def set_events(self):
        """Set the user events of the entity (events that are set during the run loop from the user)"""
        for k, v in self.user_events.items():
            setattr(self.config, k, v)
        self.user_events = {}

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the entity

        :param routine_fn: routine_fn
        :param name: routine name, defaults to None
        :param interval: routine execution interval, defaults to 1
        """
        self.routine_handler.attach_routine(routine_fn, name, interval)

    def detach_routine(self, name):
        """Detach a routine from the entity

        :param name: routine name
        """
        self.routine_handler.detach_routine(name)

    def detach_all_routines(self):
        """Detach all routines from the entity"""
        self.routine_handler.detach_all_routines()

    def routine_step(self, time, catch_errors):
        """Execute the entity's routines with their corresponding execution intervals"""
        # Give self object as parameter to the routine function so it executes functions on the entity
        self.routine_handler.routine_step(self, time, catch_errors)

    def print_infos(self):
        """Print the entity's infos

        :return: entity's infos
        """
        dict_infos = self.config.to_dict()

        info_lines = []
        info_lines.append("Entity Overview:")
        info_lines.append(f"{'-' * 20}")
        info_lines.append(f"Type: {self.etype.name}")
        info_lines.append(f"Subtype: {self.subtype_label}")
        info_lines.append(f"Idx: {self.idx}")
        info_lines.append(f"Exists: {self.exists}")
        info_lines.append(
            f"Position: x={dict_infos['x_position']:.2f}, y={dict_infos['y_position']:.2f}"
        )
        info_lines.append(f"Diameter: {self.diameter:.2f}")
        info_lines.append(f"Color: {self.color}")
        info_lines.append("")

        return print("\n".join(info_lines))

    def print_routines(self):
        """Print the entity's routines"""
        self.routine_handler.print_routines()


class Agent(Entity):
    """Agent class that represents an agent in the simulation

    :param Entity: Entity
    """

    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.AGENT
        self.behavior_handler = BehaviorHandler()
        self.logger = Logger()
        self.eating_range = 10
        self.diet = []
        self.ate = False
        self.time_since_feeding = np.inf
        self.simulation_entities = None
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
        left, right = self.config.left_prox, self.config.right_prox
        if sensed_entities is not None:
            # transform the strings of sensed entities into ints (this fn can surely be optimized)
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

    def sensed_entities(self):
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

    def sense_attributes(self, sensed_attribute, default_value=None):
        """Return the sensed attribute of the left and right sensed entities

        :param sensed_attribute: sensed_attribute
        :param default_value: default value if the attribute is not found, defaults to None
        :return: sensed attributes
        """
        left_ent, right_ent = self.sensed_entities()
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

    def print_behaviors(self):
        """Print the behaviors and active behaviors of the agent"""
        self.behavior_handler.print_behaviors()

    def change_behavior_weight(self, name, new_weight):
        """Change the weight of a behavior of the agent

        :param name: behavior name
        :param new_weight: new weight of the behavior
        """
        self.behavior_handler.change_behavior_weight(name, new_weight)

    def behave(self, time):
        """Make the agent behave according to its active behaviors

        :param time: time
        """
        self.behavior_handler.behave(self, time)
        # increment time since last meal for all alive agents
        self.time_since_feeding += 1

    def stop_motors(self):
        """Stop the motors of the agent"""
        self.left_motor = 0
        self.right_motor = 0

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


class Object(Entity):
    """Object class that represents an object in the simulation

    :param Entity: Entity
    """

    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


# mappy the entity type to their corresponding class
etype_to_class = {EntityType.AGENT: Agent, EntityType.OBJECT: Object}


class NotebookController(SimulatorController):
    """NotebookController class that enables the user to control the simulation from a notebook

    :param SimulatorController: SimulatorController
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.all_entities = []
        self.time = 0

        # add agents and objects to the global entities list
        for etype in list(EntityType):
            setattr(
                self,
                f"{etype.name.lower()}s",
                [etype_to_class[etype](c) for c in self.configs[etype.to_state_type()]],
            )
            self.all_entities.extend(getattr(self, f"{etype.name.lower()}s"))

        # set flags
        self.from_stream = True
        self._is_running = False

        # set frequency of the simulator to max speed
        self.configs[StateType.SIMULATOR][0].freq = -1

        # handle the different subtypes labels objects
        self._subtype_idx_to_label = self.subtypes_labels
        self._subtype_label_to_idx = {
            v: k for k, v in self._subtype_idx_to_label.items()
        }
        self.valid_subtypes = set(self._subtype_label_to_idx.keys())

        # add a routine handler to the controller
        self.routine_handler = RoutineHandler()

        # automatically update attributes of all entities
        self.set_all_user_events()
        self.update_agents_attributes()
        self.update_entities_attributes()

    def is_running(self):
        """Check if the simulator is running"""
        return self._is_running

    def update_agents_attributes(self):
        """Give all agents the list of all entities in the simulation (usefull for advanced cases like eating)"""
        for agent in self.agents:
            agent.simulation_entities = self.all_entities
            agent._subtype_label_to_idx = self._subtype_label_to_idx
            agent.valid_subtypes = self.valid_subtypes

    def update_entities_attributes(self):
        """Temporary fn to give their subtype labels to all entities"""
        for ent in self.all_entities:
            ent.subtype_label = self._subtype_idx_to_label[ent.subtype]

    # TODO : Clean mechanism to clean entity apparition (at seems like the entity is moving from a position to another), maybe add a time.sleep() --> LOW PRIORITY
    def spawn_entity(self, entity_idx, position=None):
        """Spawn an entity at a given position

        :param entity_idx: entity_idx
        :param position: position, defaults to None
        """
        entity = self.all_entities[entity_idx]
        try:
            if entity.exists:
                lg.warning(f"Entity {entity_idx} already exists")
                return
            if position is not None:
                entity.x_position = float(position[0])
                entity.y_position = float(position[1])
            entity.exists = True
            lg.info(
                f"Entity {entity_idx} spawned at {entity.x_position, entity.y_position}"
            )
            return entity
        except Exception as e:
            lg.error(f"Error while spawning entity {entity_idx}: {e}")

    def remove_entity(self, entity_idx):
        """Remove an entity

        :param entity_idx: entity_idx
        """
        entity = self.all_entities[entity_idx]
        if not entity.exists:
            lg.warning(f"Entity {entity_idx} already removed")
        entity.exists = False

    def remove_entity_type(self, entity_type):
        """Remove all entities of a given type

        :param entity_type: entity_type
        """
        entity_type_idx = self.get_idx_from_label_subtype(entity_type)
        for ent in self.all_entities:
            if ent.subtype == entity_type_idx:
                self.remove_entity(ent.idx)

    def start_entity_apparition(
        self, interval=50, entity_type: str = None, position_range=None
    ):
        """Start the apparition process for entities of type entity_type every period seconds

        :param interval: execution interval, defaults to 50
        :param entity_type: entity_type, defaults to None
        :param position_range: position range where entities can spawn, defaults to None
        """
        entity_type_idx = self.get_idx_from_label_subtype(entity_type)

        routine_fn = functools.partial(
            spawn_entity_routine,
            entity_type=entity_type_idx,
            position_range=position_range,
        )
        # add the name of the spawning routine function otherwise error in the routine handler
        self.attach_routine(
            routine_fn, name=spawn_entity_routine.__name__, interval=interval
        )

    def start_resources_apparition(self, interval=50, position_range=None):
        """Start the resources apparition process

        :param interval: execution interval, defaults to 5
        :param position_range: position_range, defaults to None
        """
        resources_type = "resources"
        self.start_entity_apparition(
            interval, entity_type=resources_type, position_range=position_range
        )

    def start_eating_mechanism(self, interval=10, proximeters_mode=False):
        """Start the eating mechanism for all agents

        :param interval: execution interval, defaults to 10
        :param proximeters_mode: wether to only eat entities sensed by proximeters or not, defaults to False
        """
        eating_routine = (
            eating_routine_proximeters if proximeters_mode else eating_routine_range
        )
        self.attach_routine(eating_routine, interval=interval)

    def stop_resources_apparition(self):
        """Stop the resources apparition process"""
        if spawn_entity_routine.__name__ in self.routine_handler._routines:
            self.detach_routine(spawn_entity_routine.__name__)
        else:
            lg.warning("Resources apparition is already stopped")

    def stop_eating_mechanism(self):
        if eating_routine_range.__name__ in self.routine_handler._routines:
            self.detach_routine(eating_routine_range.__name__)
        elif eating_routine_proximeters.__name__ in self.routine_handler._routines:
            self.detach_routine(eating_routine_proximeters.__name__)
        else:
            lg.warning("Eating mechanism is already stopped")

    def set_all_user_events(self):
        """Set all user events from clients (interface or notebooks) for all entities"""
        for e in self.all_entities:
            e.set_events()

    def run(self, threaded=True, num_steps=math.inf, debug_mode=False):
        """Run the simulation

        :param threaded: wether to run the simulation in a thread or not, defaults to True
        :param num_steps: num_steps, defaults to math.inf
        :raises RuntimeError: if the simulator is already started
        """
        # automatically catch errors only if not in debug mode
        catch_errors = not debug_mode
        if self._is_running:
            print("Simulator is already started")
            return
        self._is_running = True
        if threaded:
            run_thread = threading.Thread(
                target=self._run, args=(num_steps, catch_errors)
            )
            run_thread.daemon = True
            run_thread.start()
        else:
            self._run(num_steps=num_steps, catch_errors=catch_errors)

    def _run(self, num_steps=math.inf, catch_errors=True):
        """run the simulation for a given number of steps

        :param num_steps: num_steps, defaults to math.inf
        :param catch_errors: wether to catch errors or not, defaults to False
        """
        # Add a local time for the run function independant from the controller time
        run_time = 0
        while run_time < num_steps and self._is_running:
            with self.batch_set_state():
                # execute routines of the controller
                self.controller_routine_step(self.time, catch_errors=catch_errors)

                # execute routines of the existing entities
                for entity in self.all_entities:
                    entity.set_events()
                    if not entity.exists:
                        continue
                    entity.routine_step(self.time, catch_errors=catch_errors)
                    # execute behaviors of agents
                    if entity.etype == EntityType.AGENT:
                        entity.behave(self.time)

            # update the attributes of all entities and do a step on server side
            self.state = self.client.step()
            self.pull_configs()
            self.time += 1
            run_time += 1

        # finally stop the simulation
        self.stop()

    def stop(self):
        """Pause the simulation"""
        if not self._is_running:
            print("Simulator is already stopped")
        self._is_running = False

    def wait(self, seconds):
        """Wait for a given number of seconds

        :param seconds: seconds
        """
        time.sleep(seconds)

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the simulator

        :param routine_fn: routine_fn
        :param name: routine name, defaults to None
        """
        self.routine_handler.attach_routine(routine_fn, name, interval)

    def detach_routine(self, name):
        """Detach a routine from the entity

        :param name: routine name
        """
        self.routine_handler.detach_routine(name)

    def detach_all_routines(self):
        """Detach all routines from the entity"""
        self.routine_handler.detach_all_routines()

    def controller_routine_step(self, time, catch_errors):
        """Execute the simulator routines"""
        self.routine_handler.routine_step(self, time, catch_errors)

    def print_subtypes_list(self):
        """Return the list of subtypes

        :return: subtypes list
        """
        print(list(self.subtypes_labels.values()))

    def get_idx_from_label_subtype(self, label):
        """Return the index of a subtype from its label

        :param label: label
        :return: index of the subtype
        """
        assert (
            label in self.valid_subtypes
        ), f"Please specify a valid entity type among {self.valid_subtypes}"
        entity_type_idx = self._subtype_label_to_idx[label]
        return entity_type_idx

    def print_fps(self, record_time=2, server=False):
        """Compute the fps of the simulation for a given record time without blocking

        :param record_time: record_time, defaults to 2
        :param server_time: wether to record steps per seconds in the server or in the controller, defaults to False
        """
        print(
            f"measuring the FPS (number of steps per second) in the {'server' if server else 'controller'} during {record_time} seconds ..."
        )
        start_time = self.time if not server else self.server_time

        def calculate_fps():
            time.sleep(record_time)
            end_time = self.time if not server else self.server_time
            fps = (end_time - start_time) / record_time
            print(f"FPS: {fps:.2f}")

        # use a thread to calculate the fps without blocking the run loop
        threading.Thread(target=calculate_fps).start()

    def print_routines(self):
        """Print the controller's routines"""
        self.routine_handler.print_routines()

    @property
    def server_time(self):
        """Return the current time of the simulation

        :return: time
        """
        return self.configs[StateType.SIMULATOR][0].time

    @property
    def box_size(self):
        """Return the box size of the simulation

        :return: box size
        """
        return self.configs[StateType.SIMULATOR][0].box_size

    @property
    def existing_agents(self):
        """Return the list of existing agents

        :return: existing agents
        """
        return [agent for agent in self.agents if agent.exists]

    @property
    def non_existing_agents(self):
        """Return the list of non existing agents

        :return: non existing agents
        """
        return [agent for agent in self.agents if not agent.exists]


# Predefined routines that can be attached to the controller


def spawn_entity_routine(controller, entity_type=None, position_range=None):
    """Spawn entities of type entity_type every period seconds within a given position range

    :param period: period
    :param entity_type: entity_type, defaults to None
    :param position_range: position_range, defaults to None
    """
    assert entity_type is not None, "Please specify the entity type"
    assert isinstance(entity_type, int), "Entity type must be an integer index"

    # transform the position range if not specified
    if position_range is None:
        position_range = ((0, controller.box_size), (0, controller.box_size))

    non_existing_ent_list = [
        ent.idx
        for ent in controller.all_entities
        if not ent.exists and ent.subtype == entity_type
    ]
    if non_existing_ent_list:
        ent_idx = np.random.choice(non_existing_ent_list)
        x = np.random.uniform(position_range[0][0], position_range[0][1], 1)
        y = np.random.uniform(position_range[1][0], position_range[1][1], 1)
        controller.spawn_entity(ent_idx, position=(x, y))
    else:
        lg.info(f"All entities of type {entity_type} are spawned")


def eating_routine_range(controller):
    """Make agents eat entities if they are in their diet and eating range

    :param controller: NotebookController
    """
    for agent in controller.existing_agents:
        for entity_type in agent.diet:
            assert (
                entity_type in controller.valid_subtypes
            ), f"Please specify a valid entity type among {controller.valid_subtypes}, for agent {agent.idx} diet : {agent.diet} "
            # transform the entity type label into an idx
            # TODO : use this fn get_idx_from_label_subtype instead of the list here (test it works well)
            entity_type = controller._subtype_label_to_idx[entity_type]
            # get the idx of entities that are eatable by the agent (by precaution remove the agent itself)
            eatable_entities_idx = [
                ent.idx
                for ent in controller.all_entities
                if ent.subtype == entity_type and ent.idx != agent.idx
            ]
            distances = agent.config.proximity_map_dist[eatable_entities_idx]
            in_range = distances < agent.eating_range
            # arr_idx is the index of the in_range array
            for arr_idx, ent_idx in enumerate(eatable_entities_idx):
                if in_range[arr_idx] and controller.all_entities[ent_idx].exists:
                    controller.remove_entity(ent_idx)
                    agent.ate = True
                    agent.time_since_feeding = 0


def eating_routine_proximeters(controller):
    """Make agents eat entities if they are in their diet, eating range and sensed by their proximeters

    :param controller: NotebookController
    """
    for agent in controller.existing_agents:
        left_prox, right_prox = agent.sensors()
        left_type_idx, right_type_idx = agent.prox_sensed_ent_type
        # TODO : could improve this step by also directly storing the diet as a list of idx ijnstead of computing it each time
        diet_idx = [
            controller.get_idx_from_label_subtype(entity) for entity in agent.diet
        ]

        can_eat_left = (
            left_type_idx in diet_idx
            and (1.0 - left_prox) * agent.proxs_dist_max <= agent.eating_range
        )
        can_eat_right = (
            right_type_idx in diet_idx
            and (1.0 - right_prox) * agent.proxs_dist_max <= agent.eating_range
        )

        # if the agent can eat
        if can_eat_left or can_eat_right:
            # determine which side to eat
            if can_eat_left and can_eat_right:
                eating_choice = np.random.choice([0, 1])
            else:
                eating_choice = 0 if can_eat_left else 1

            controller.remove_entity(agent.prox_sensed_ent_idx[eating_choice])
            agent.ate = True
            agent.time_since_feeding = 0
        else:
            agent.ate = False
