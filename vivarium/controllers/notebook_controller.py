import time
import math
import logging
import warnings
import threading
import functools

import numpy as np

from vivarium.controllers.utils import RoutineHandler
from vivarium.utils.scene_configs import load_scene_config
from vivarium.controllers.vivarium_controller import VivariumController
from vivarium.utils.handle_server_interface import start_server_and_interface, stop_server_and_interface




lg = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")


if logging.root.handlers:
    lg.setLevel(logging.root.level)
else:
    lg.setLevel(logging.WARNING)

   
class NotebookController(VivariumController):
    """
    NotebookController class that enables the user to control the simulation on the client side, typically from a Jupyter Notebook
    """
    def __init__(self, client=None, subtypes=[], **controllers):
        
        super().__init__(client=client, subtypes=subtypes, **controllers)
        self.time = 0

        self._is_running = False

        # # set frequency of the simulator to max speed
        self.client.freq = -1

        # handle the different subtypes labels objects
        self._subtype_idx_to_label = self.subtype_labels
        self._subtype_label_to_idx = {
            v: k for k, v in self._subtype_idx_to_label.items()
        }
        self.valid_subtypes = set(self._subtype_label_to_idx.keys())

        # add a routine handler to the controller
        self.routine_handler = RoutineHandler()

    @classmethod
    def from_config(cls, config, client=None, notebook_control=True):
        return super().from_config(config, client=client, notebook_control=notebook_control)

    def is_running(self):
        """Check if the simulator is running"""
        return self._is_running

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
            self.execute_routines_and_behaviors(catch_errors=catch_errors)

            self.step()

            self.time += 1
            run_time += 1

        # finally stop the simulation
        self.stop()

    def execute_routines_and_behaviors(self, catch_errors=True):
        # execute routines of the controller
        self.controller_routine_step(self.time, catch_errors=catch_errors)

        # execute routines of the existing entities
        for _, controller in self.controllers.items():
            controller.step(self.time, catch_errors=catch_errors)
            
            # # TODO : Add a check to ensure that the entity exists
            # for entity in elist:
            #     entity.step(self.time, catch_errors=catch_errors)

    def stop(self):
        """Pause the simulation"""
        if not self._is_running:
            print("Simulator is already stopped")
        self._is_running = False

    def stop_session(self, safe_mode=False):
        """Stop the session: simulation, server and interface"""
        if self._is_running:
            self.stop()
        stop_server_and_interface(safe_mode=safe_mode)

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

    # TODO: this method to be revised
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

    # @property
    # def server_time(self):
    #     """Return the current time of the simulation

    #     :return: time
    #     """
    #     return self.configs[StateType.SIMULATOR][0].time

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


# TODO: Remove the routines below once the server-side versions of them are fully working?
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
