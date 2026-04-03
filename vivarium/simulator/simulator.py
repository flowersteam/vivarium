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
from dataclasses import dataclass, is_dataclass
from omegaconf.errors import ConfigKeyError, ConfigAttributeError, InterpolationKeyError

from vivarium.utils.dataclass_wrapper import (
    update_dataclass_from_change_list, create_dataclass_from_dict, Remote
)
from vivarium.utils.scene_configs import extend_controller_kwargs

from vivarium.utils.timer import SleepTimer, sleep_timer

lg = logging.getLogger(__name__)
# lg.setLevel(logging.DEBUG)


def update_from_dataclass(obj, dataclass_instance, exclude_fields=[]):
    for field in dataclass_instance.__dataclass_fields__.keys():
        if field not in exclude_fields:
            if is_dataclass(getattr(dataclass_instance, field)):
                setattr(obj, field, update_from_dataclass(getattr(obj, field), getattr(dataclass_instance, field), exclude_fields))
            else:
                setattr(obj, field, getattr(dataclass_instance, field))
    return obj


nested_fields_to_access = { # Now unused?
    'env': [
        'box_size',
        'neighbor_radius',
        'num_scan_steps',
        'to_jit'
    ]
}

@dataclass
class StateAndControllerParameters:
    state: any
    controller_parameters: any


class Simulator:
    def __init__(self, env, state=None, controller_parameters=None, scene_name=None, freq=-1):
        self.env = env
        self.controller_parameters = controller_parameters

        self.state = state or env.init_state()
        
        self._freq = freq
        self.sleep_timer = SleepTimer(freq)        
        self._is_running = False
        self._to_stop = False
        self._was_running = False
        self.name = 'server'
        self.remote = Remote(self)

        # Attributes to record simulation (probably broken for now)
        self.recording = False
        self.records = None
        self.saving_dir = None
        
        self.is_grpc_client = False
        
        lg.info("Simulator initialized")
        
    @classmethod
    def from_config(cls, simulator_config):
        kwargs = {}
        if 'client' in simulator_config and 'controller_kwargs' in simulator_config.client:
            kwargs['simulator'] = OmegaConf.to_container(simulator_config.client.controller_kwargs, resolve=True)
        for name, c_config in simulator_config.env.components.component_list.items():
            if 'client' in c_config and 'controller_kwargs' in c_config.client:
                if 'n_max' in c_config:
                    n_max = c_config.n_max
                    controller_kwargs = extend_controller_kwargs(c_config.client.controller_kwargs, getattr(c_config, 'by_indices', []), n_max)
                else:
                    controller_kwargs = c_config.client.controller_kwargs
                kwargs[name] = OmegaConf.to_container(controller_kwargs, resolve=True)

        cp = create_dataclass_from_dict('ControllerParameters', kwargs)

        try:
            scene_name = simulator_config.client.controller_kwargs.scene_name
        except (ConfigKeyError, ConfigAttributeError, InterpolationKeyError):
            logging.warning("Scene name not found, Simulator.scene_name will be None.")
            scene_name = None
        
        if 'client' in simulator_config and 'controller_kwargs' in simulator_config.client and 'freq' in simulator_config.client.controller_kwargs:
            freq = simulator_config.client.controller_kwargs.freq 
        else:
            freq = -1
        
        return cls(env=hydra.utils.get_class(simulator_config.env._target_).from_config(simulator_config.env),
                   controller_parameters=cp,
                   scene_name=scene_name,
                   freq=freq
                  )
    
    def to_config(self, state):
        
        with hydra.initialize(config_path='../../conf/scene/simulator', version_base=None):
            cfg = hydra.compose(config_name="base_simulator")
            cfg = OmegaConf.merge(
                cfg, 
                OmegaConf.create(
                    {'client': {
                        'controller_kwargs': {
                            'subtype_labels': self.controller_parameters.simulator.subtype_labels,
                            'freq': self.freq,
                            'scene_name': self.scene_name
                            }
                        },
                     'env': self.env.to_config(state)
                     }
            ))
        return cfg
    
    @property
    def scene_name(self):
        return self.controller_parameters.simulator.scene_name

    @property
    def freq(self):
        return self._freq
    
    @freq.setter
    def freq(self, value):
        self._freq = value
        self.sleep_timer.frequency = value

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

        return new_state

    def step(self, changes=None):
        
        if changes is not None and len(changes) > 0:
            self.set_changes(changes)
        self.state = self._step(self.state)
        
        # record the env state because it is the one we can plot and use without client-server interaction
        if self.recording:
            self.record(self.state)
            
        if changes is None:
            return self.state
        else:
            return StateAndControllerParameters(state=self.state, controller_parameters=self.controller_parameters)

    def run(self, threaded=False, num_steps=math.inf, save=False, saving_name=None):
        """Run the simulator for the desired number of timesteps, either in a separate thread or not. Return the final state

        :param threaded: wether to run the simulation in a thread or not, defaults to False
        :param num_steps: number of step loops before stopping the simulation run, defaults to math.inf
        :raises ValueError: raise an error if the simulator is already running
        """
        # Check is the simulator isn't already running
        if self._is_running:
            raise ValueError("Simulator is already runnning")
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
        lg.debug("Starting simulator _run")
        self._is_running = True
        lg.info("Simulation run starts")

        loop_count = 0
        
        if save:
            self.start_recording(saving_name)

        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            with sleep_timer(timer=self.sleep_timer):
            
                if self._to_stop:
                    lg.debug("Stopping simulator _run as requested")
                    self._to_stop = False
                    break

                self.step()
                loop_count += 1
            
        if save:
            self.stop_recording()

        self._is_running = False
        lg.info("Simulation run stops")

    def is_running(self):
        return self._is_running or self._was_running

    def set_changes(self, changes, update_from_server=True):
        # update_from_server is only here to match the SimulatorGRPCClient interface
        
        lg.debug("Applying changes to simulator")
        lg.debug(f"Changes: {changes}")
        self = update_dataclass_from_change_list(self, changes)

        self = update_from_dataclass(self, self.controller_parameters.simulator, exclude_fields=['scene_name'])
        
        if self.run_from == self.name:
            if self.is_running() != self.simulation_running:
                if self.simulation_running:
                    lg.debug("Starting simulator from server apply_changes")
                    self.run(threaded=True)
                else:
                    lg.debug("Stopping simulator from server apply_changes")
                    self.stop()
        elif self.is_running():
            lg.debug("Stopping simulator from server apply_changes (2nd case)")
            self.stop()                    

        return self.controller_parameters

    def get_state(self):
        """Get current simulation state

        :return: simulation state
        """
        return self.state

    def get_controller_parameters(self):
        return self.controller_parameters
    
    def get_state_and_controller_parameters(self):
        return StateAndControllerParameters(state=self.state, controller_parameters=self.controller_parameters)

    def stop(self, blocking=True):
        """Stop the simulation

        :param blocking: If True, wait for the simulation to actually stop before returning
        """
        if self._is_running:
            self._to_stop = True
        if blocking:
            while self._is_running:
                time.sleep(0.01)
                lg.info("still running")
            lg.info("now stopped")
            
    def register_client(self, client_name):
        if client_name not in self.controller_parameters.simulator.client_names:
            self.controller_parameters.simulator.client_names.append(client_name)
            lg.info(f"Client {client_name} registered to simulator.")
        else:
            lg.warning(f"Client {client_name} is already registered.")

    def unregister_client(self, client_name):
        if client_name in self.controller_parameters.simulator.client_names:
            self.controller_parameters.simulator.client_names.remove(client_name)
            lg.info(f"Client {client_name} unregistered from simulator.")
        else:
            lg.warning(f"Client {client_name} is not registered.")

    @contextmanager
    def pause(self):
        """Pause the simulation

        :yield: dummy self
        """
        lg.debug("Pausing simulator")
        self._was_running = self.is_running()
        lg.debug(f"Was running: {self._was_running}")
        if self._was_running:
            self.stop(blocking=True)
        try:
            yield self
        finally:
            lg.debug(f"Resuming simulator: was_running={self._was_running}, simulation_running={self.simulation_running}, _is_running={self._is_running}")
            if self._was_running and self.simulation_running and not self._is_running:
                lg.debug("Running simulator from pause context manager")
                self.run(threaded=True)
            self._was_running = False
                
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
