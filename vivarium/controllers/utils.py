import logging
import threading

import numpy as np

lg = logging.getLogger(__name__)


class Logger(object):
    def __init__(self):
        """Logger class that logs data for the agents"""
        self.logs = {}

    def add(self, log_field, data):
        """Add data to the log_field of the logger

        :param log_field: log_field
        :param data: data
        """
        if log_field not in self.logs:
            self.logs[log_field] = [data]
        else:
            self.logs[log_field].append(data)

    def get_log(self, log_field):
        """Get the log of the logger for a specific log_field

        :param log_field: log_field
        :return: data associated with the log_field
        """
        if log_field not in self.logs:
            print("No topic called " + log_field)
            return []
        else:
            return self.logs[log_field]

    def clear(self):
        """Clear all logs of the logger"""
        del self.logs
        self.logs = {}


class RoutineHandler(object):
    """RoutineHandler class that handles routines for the NotebookController and its Entities"""

    def __init__(self):
        self._routines = {}
        self._active_routines = {}
        self._lock = threading.Lock()

    def attach_routine(self, routine_fn, name=None, interval=1, start=True):
        """Attach a routine to the entity

        :param routine_fn: routine function
        :param name: routine name, defaults to None (routine function name)
        :param interval: routine execution interval, defaults to 1
        :param start: whether to start the routine, defaults to True
        """
        assert (
            isinstance(interval, int) and interval > 0
        ), "Interval must be a positive integer"
        with self._lock:
            self._routines[name or routine_fn.__name__] = (routine_fn, interval)
        if start:
            self.start_routine(name or routine_fn.__name__)

    def start_routine(self, routine_fn):
        """Start a routine of the entity

        :param routine_fn: routine function or its name as a string
        """
        with self._lock:
            name = routine_fn.__name__ if hasattr(routine_fn, "__name__") else routine_fn
            self._active_routines[name] = self._routines[name]

    def stop_routine(self, routine_fn):
        """Stop a routine of the entity

         :param routine_fn: routine function or its name as a string
        """
        with self._lock:
            name = routine_fn.__name__ if hasattr(routine_fn, "__name__") else routine_fn
            if name in self._active_routines:
                del self._active_routines[name]
            else:
                lg.warning(f"Routine {name} is not started")

    def detach_routine(self, routine_fn):
        """Detach a routine from the entity

         :param routine_fn: routine function or its name as a string
        """
        with self._lock:
            name = routine_fn.__name__ if hasattr(routine_fn, "__name__") else routine_fn
            if name in self._routines:
                del self._routines[name]
            else:
                lg.warning(f"Routine {name} is not attached")
            if name in self._active_routines:
                del self._active_routines[name]

    def detach_all_routines(self):
        """Detach all routines from the entity"""
        for name in list(self._routines.keys()):
            self.detach_routine(name)

    def routine_step(self, entity, time, catch_errors=True):
        """Execute the entity's routines with their corresponding execution intervals, and remove the ones that cause errors"""
        to_remove = []
        with self._lock:
            # iterate over the routines and check if they work
            for name, (fn, interval) in self._active_routines.items():
                if time % interval == 0:
                    # if the catch_errors flag is set to True, catch the errors and remove the routine if it fails
                    if catch_errors:
                        try:
                            # execute the function on the entity object if the routine works
                            fn(entity)
                        except Exception as e:
                            # else plot an error message and remove the routine
                            lg.error(
                                f"Error while executing routine: {e}, removing routine {name}"
                            )
                            to_remove.append(name)
                    # else just execute the routines normally
                    else:
                        fn(entity)

        # remove all problematic routines at the end to prevent spamming error messages and crashing the program
        if catch_errors:
            for name in to_remove:
                self.detach_routine(name)

    def print_routines(self):
        """Print the behaviors and active behaviors of the agent"""
        with self._lock:
            if len(self._routines) == 0:
                print("No routines attached")
            else:
                available_routines = list(self._routines.keys())
                active_routines = list(self._active_routines.keys())
                print(
                    f"Available routines: {available_routines}, Active routines: {active_routines if active_routines else 'No active behaviors'}"
                )


class BehaviorHandler(object):
    """BehaviorHandler class that handles behaviors for agents"""

    def __init__(self):
        """Initialize the BehaviorHandler"""
        self._behaviors = {}
        self._started_behaviors = {}
        self._lock = threading.Lock()

    def attach_behavior(
        self, behavior_fn, name=None, interval=1, weight=1.0, start=True
    ):
        """Attach a behavior to the agent with a given weight

        :param behavior_fn: behavior function
        :param name: behavior name, defaults to None (behavior function name)
        :param interval: behavior execution interval, defaults to 1
        :param weight: behavior weight, defaults to 1.
        :param start: whether to start the behavior, defaults to True
        """
        assert (
            isinstance(interval, int) and interval > 0
        ), "Interval must be a positive integer"
        assert (
            isinstance(weight, (int, float)) and weight >= 0
        ), "Weight must be a positive number"
        with self._lock:
            self._behaviors[name or behavior_fn.__name__] = [
                behavior_fn,
                interval,
                weight,
            ]
        if start:
            self.start_behavior(name or behavior_fn.__name__)

    def detach_behavior(self, behavior_fn):
        """Detach a behavior from the agent

        :param behavior_fn: behavior function or its name as a string
        """
        with self._lock:
            name = behavior_fn.__name__ if hasattr(behavior_fn, "__name__") else behavior_fn
            if name in self._behaviors:
                del self._behaviors[name]
            else:
                lg.warning(f"Behavior {name} is not attached")
            if name in self._started_behaviors:
                del self._started_behaviors[name]

    def detach_all_behaviors(self):
        """Detach all behaviors from the agent"""
        for name in list(self._behaviors.keys()):
            self.detach_behavior(name)

    def start_behavior(self, behavior_fn):
        """Start a behavior of the agent

        :param behavior_fn: behavior function or its name as a string
        """
        with self._lock:
            name = behavior_fn.__name__ if hasattr(behavior_fn, "__name__") else behavior_fn
            self._started_behaviors[name] = self._behaviors[name]

    def start_all_behaviors(self):
        """Start all behaviors of the agent"""
        with self._lock:
            for name in self._behaviors:
                self._started_behaviors[name] = self._behaviors[name]

    def stop_behavior(self, behavior_fn):
        """Stop a behavior of the agent

        :param behavior_fn: behavior function or its name as a string
        """
        with self._lock:
            name = behavior_fn.__name__ if hasattr(behavior_fn, "__name__") else behavior_fn
            if name in self._started_behaviors:
                del self._started_behaviors[name]
            else:
                lg.warning(f"Behavior {name} is not started")

    def change_behavior_weight(self, behavior_fn, new_weight):
        """Change the weight of a behavior

        :param behavior_fn: behavior function or its name as a string
        :param new_weight: new weight
        """
        assert (
            isinstance(new_weight, (int, float)) and new_weight >= 0
        ), "New weight must be a positive number"
        with self._lock:
            name = behavior_fn.__name__ if hasattr(behavior_fn, "__name__") else behavior_fn
            # update the weight of the behavior in the behaviors and started_behaviors tuples
            self._behaviors[name][2] = new_weight
            if name in self._started_behaviors:
                self._started_behaviors[name][2] = new_weight

    def behave(self, agent, time):
        """Make the agent behave according to its active behaviors

        :param time: current time
        """
        if not self._started_behaviors:
            return

        # prevents simulator to crash if an error occurs during motor values computations
        try:
            motor_contributions = [
                (w * np.array(fn(agent)), w)
                for (fn, interval, w) in self._started_behaviors.values()
                if time % interval == 0
            ]
            # check if there are motor contributions from an active behavior (it can be empty because of the interval)
            if motor_contributions:
                total_motor_values = sum(
                    motor_value for (motor_value, _) in motor_contributions
                )
                total_weights = sum(weight for (_, weight) in motor_contributions)
                if total_weights == 0:
                    motor_values = np.zeros(2)
                else:
                    motor_values = total_motor_values / total_weights
            # else keep the current motor values
            else:
                motor_values = [agent.left_motor, agent.right_motor]
        except Exception as e:
            lg.error(
                f"Error while computing motor values in behavior of agent {agent.idx}: {e}"
            )
            motor_values = np.zeros(2)

        agent.left_motor, agent.right_motor = motor_values

    def print_behaviors(self, full_infos=False):
        """Print the behaviors and active behaviors of the agent"""
        with self._lock:
            if len(self._behaviors) == 0:
                print("No behavior attached")
            else:
                attached_behaviors = list(self._behaviors.keys())
                started_behaviors = list(self._started_behaviors.keys())
                started_print = (
                    "No behavior started"
                    if not started_behaviors
                    else f"Started behaviors: {started_behaviors}"
                )
                print(f"Attached behaviors: {attached_behaviors}, {started_print}")
                if full_infos:
                    for name, (fn, interval, weight) in self._behaviors.items():
                        print(f"Behavior {name}: interval={interval}, weight={weight}")
