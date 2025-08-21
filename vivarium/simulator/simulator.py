import os
import time
import math
import hydra
import pickle
import logging
import datetime
import threading
from functools import partial
from omegaconf import OmegaConf
from contextlib import contextmanager
from omegaconf.errors import ConfigKeyError, ConfigAttributeError, InterpolationKeyError

from vivarium.utils.scene_configs import extend_kwargs
from vivarium.utils.converters import access_nested_fields
from vivarium.simulator.config import SimulatorConfiguration
from vivarium.controllers.dataclass_wrapper import update_dataclass_from_change_list, create_dataclass_from_dict


lg = logging.getLogger(__name__)


nested_fields_to_access = {
    'env': [
        'box_size',
        'neighbor_radius',
        'num_scan_steps',
        'to_jit'
    ]
}


@access_nested_fields(nested_fields_to_access)
class Simulator:
    def __init__(self, env, state=None, controller_parameters=None, scene_name=None, freq=-1):
        
        self.env = env
        self.controller_parameters = controller_parameters
        self.scene_name = scene_name
        self.state = state or env.init_state()
        self.freq = freq
        self._is_started = False
        self._to_stop = False

        # Attributes to record simulation
        self.recording = False
        self.records = None
        self.saving_dir = None

        lg.info("Simulator initialized")

    @classmethod
    def from_config(cls, config):

        try:
            kwargs = {}
            for etype, c_config in config.clients.items():
                n_max = config.env.components.component_list[etype].n_max
                controller_kwargs = extend_kwargs(c_config.controller_kwargs, n_max)
                kwargs[etype] = OmegaConf.to_container(controller_kwargs, resolve=True)
            cp = create_dataclass_from_dict('ControllerParameters', kwargs)
        except (ConfigKeyError, ConfigAttributeError, InterpolationKeyError):
            logging.warning("Client configuration not found, Simulator.controller_parameters will be None.")
            cp = None

        try:
            scene_name = config.scene_name
        except (ConfigKeyError, ConfigAttributeError, InterpolationKeyError):
            logging.warning("Scene name not found, Simulator.scene_name will be None.")
            scene_name = None
        
        return cls(
            env=hydra.utils.get_class(config.env._target_).from_config(config.env),
            freq=config.freq,
            scene_name=scene_name,
            controller_parameters=cp
        )
    
    def to_config(self, state):
        
        with hydra.initialize(config_path='../../conf/scene/simulator', version_base=None):
            cfg = hydra.compose(config_name="base_simulator")
            cfg = OmegaConf.merge(cfg, OmegaConf.create({
                'freq': self.freq,
                'env': self.env.to_config(state)
            }))
        return cfg

    # def load_scene(self, scene_name):
    #     """Load a scene in the simulator

    #     :param scene_name: scene to load
    #     """
    #     lg.info("Loading a new scene\n")

    #     if self.is_started():
    #         self.stop(blocking=True)
    #     scene_config = SceneConfiguration(scene_name=scene_name)
    #     self.freq = scene_config.config.simulator.kwargs.freq
    #     del self.env
    #     self.env = scene_config.create_environment()
    #     self.state = self.env.state

    def init_state(self):
        if self.state.entity_state.momentum is None:
            self.state = self.env.init_fn(self.state)

    def _step(self, state):
        """Do num_updates jitted steps in the simulation. This is done by converting state into environment state, and convert it back to simulation state during return

        :param state: current simulation state
        :param num_updates: current simulation neighbors array
        :return: updated state
        """
        new_state = self.env.step(state=state)

        # record the env state because it is the one we can plot and use without client-server interaction
        if self.recording:
            self.record(new_state)

        # return the next sim state (convert new env state)
        return new_state  # self.env_to_sim_state(new_env_state)

    def step(self, changes=[]):
        """Do a step in the simulation by calling _step"""
        if len(changes) > 0:
            self.apply_changes(changes)
        self.state = self._step(self.state)
        return self.state

    def run(self, threaded=False, num_steps=math.inf, save=False, saving_name=None):
        """Run the simulator for the desired number of timesteps, either in a separate thread or not. Return the final state

        :param threaded: wether to run the simulation in a thread or not, defaults to False
        :param num_steps: number of step loops before stopping the simulation run, defaults to math.inf
        :raises ValueError: raise an error if the simulator is already running
        """
        # Check is the simulator isn't already running
        if self._is_started:
            raise ValueError("Simulator is already started")
        # Else run it either in a thread or not
        if threaded:
            # Set the _run attribute with a partial function to launch it in a thread
            _run = partial(
                self._run, num_steps=num_steps, save=save, saving_name=saving_name
            )
            threading.Thread(target=_run).start()
        else:
            self._run(num_steps=num_steps, save=save, saving_name=saving_name)

    def _run(self, num_steps, save, saving_name):
        """Function that runs the simulator for the desired number of steps. Used to be called either normally or in a thread.

        :param num_steps: number of simulation steps
        """
        self._is_started = True
        lg.info("Simulation run starts")

        loop_count = 0
        sleep_time = 0

        if save:
            self.start_recording(saving_name)

        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            start = time.time()
            if self._to_stop:
                self._to_stop = False
                break

            self.step()
            loop_count += 1

            # Sleep for updated sleep_time seconds
            end = time.time()
            sleep_time = self.update_sleep_time(
                frequency=self.freq, elapsed_time=end - start
            )
            time.sleep(sleep_time)

        if save:
            self.stop_recording()

        # Encode that the simulation isn't started anymore
        self._is_started = False
        lg.info("Simulation run stops")

    def update_sleep_time(self, frequency, elapsed_time):
        """Compute the time we need to sleep to respect the update frequency

        :param frequency: update state frequency
        :param elapsed_time: time already used to compute the state
        :return: time needed to sleep in addition to elapsed time to respect the frequency
        """
        # if we use the freq, compute the correct sleep time
        if float(frequency) > 0.0:
            perfect_time = 1.0 / float(frequency)
            sleep_time = max(perfect_time - elapsed_time, 0)
        # Else set it to zero
        else:
            sleep_time = 0
        return sleep_time

    def start_recording(self, saving_name):
        """Start the recording of the simulation
        :param saving_name: optional name of the saving file
        """
        if self.recording:
            lg.warning(
                "You called start_recording but the simulation is already being recorded"
            )
        self.recording = True
        self.records = []

        # Either create a saving_dir with the given name or one with the current datetime
        if saving_name:
            saving_dir = f"Results/{saving_name}"
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            saving_dir = f"Results/experiment_{current_time}"

        self.saving_dir = saving_dir
        # Create a saving dir if it doesn't exist yet, TODO : Add a warning if risk of overwritting already existing content
        os.makedirs(self.saving_dir, exist_ok=True)
        lg.info("Saving directory %s created", self.saving_dir)

    def record(self, data):
        """Record the desired data during a step
        :param data: saved data (e.g simulator.state)
        """
        if not self.recording:
            lg.warning("Recording not started yet.")
            return
        self.records.append(data)

    def save_records(self):
        """Save the recorded steps in a pickle file"""
        if not self.records:
            lg.warning("No records to save.")
            return

        saving_path = f"{self.saving_dir}/frames.pkl"
        with open(saving_path, "wb") as f:
            pickle.dump(self.records, f)
            lg.info("Simulation frames saved in %s", saving_path)

    def stop_recording(self):
        """Stop the recording, save the recorded steps and reset recording information"""
        if not self.recording:
            lg.warning("Recording not started yet.")
            return

        self.save_records()
        self.recording = False

    # TODO: This shouldn't be a method, just a function
    def load(self, saving_name):
        """Load data corresponding to saving_name
        :param saving_name: name used while saving the data
        :return: loaded data
        """
        saving_path = f"Results/{saving_name}/frames.pkl"
        with open(saving_path, "rb") as f:
            data = pickle.load(f)
            lg.info("Simulation loaded from %s", saving_path)
            return data

    def apply_changes(self, changes):
        self = update_dataclass_from_change_list(self, changes)

    def start(self):
        """Start the simulation"""
        self.run(threaded=True)

    def stop(self, blocking=True):
        """Stop the simulation

        :param blocking: If True, wait for the simulation to actually stop before returning
        """
        self._to_stop = True
        if blocking:
            while self._is_started:
                time.sleep(0.01)
                lg.info("still started")
            lg.info("now stopped")

    def is_started(self):
        """Check if simulation is started

        :return: True if started else False
        """
        return self._is_started

    @contextmanager
    def pause(self):
        """Pause the simulation

        :yield: dummy self
        """
        self.stop(blocking=True)
        try:
            yield self
        finally:
            self.run(threaded=True)

    def get_state(self):
        """Get current simulation state

        :return: simulation state
        """
        return self.state
    
    def get_simulator_parameters(self):
        return SimulatorConfiguration.from_simulator(self)
