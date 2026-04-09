import os
import time
import math
import hydra
import logging
import threading
from functools import partial
from omegaconf import OmegaConf
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from vivarium.utils.dataclass_wrapper import (
    update_dataclass_from_change_list, create_dataclass_from_dict, Remote
)
from vivarium.utils.scene_configs import extend_controller_kwargs

from vivarium.utils.timer import SleepTimer, sleep_timer
from vivarium.utils.runtime import get_config_dir

lg = logging.getLogger(__name__)
# lg.setLevel(logging.DEBUG)


def sync_dataclass_fields(target, source, exclude_fields=()):
    """Copy field values from a source dataclass onto a target object, recursively.

    For each field in the source dataclass, sets the corresponding attribute on
    the target. Nested dataclass fields are synced recursively rather than replaced.

    Note:
        Modifies target in place. The return value is the same object, provided
        for convenience.

    Args:
        target: The object to update (does not need to be a dataclass).
        source: A dataclass instance whose fields provide the new values.
        exclude_fields: Field names to skip.

    Returns:
        The updated target object.
    """
    for field_name in source.__dataclass_fields__:
        if field_name not in exclude_fields:
            source_value = getattr(source, field_name)
            if is_dataclass(source_value):
                sync_dataclass_fields(getattr(target, field_name), source_value, exclude_fields)
            else:
                setattr(target, field_name, source_value)
    return target


@dataclass
class StateAndControllerParameters:
    state: any
    controller_parameters: any


class Simulator:
    def __init__(self, env, state=None, controller_parameters=None, scene_name=None, freq=-1):
        self.env = env
        if controller_parameters is None:
            controller_parameters = create_dataclass_from_dict('ControllerParameters', {
                'simulator': {'freq': freq, 'scene_name': scene_name,
                             'run_from': 'server', 'simulation_running': False,
                             'client_names': []}
            })
        self.controller_parameters = controller_parameters

        self.state = state or env.init_state()

        self.sleep_timer = SleepTimer(controller_parameters.simulator.freq)
        self._is_running = False
        self._to_stop = False
        self._was_running = False
        self.name = 'server'
        self.remote = Remote(self)

        self.is_grpc_client = False

        lg.info("Simulator initialized")

    def __setattr__(self, name, value):
        # Let properties handle themselves via the descriptor protocol
        if isinstance(getattr(type(self), name, None), property):
            super().__setattr__(name, value)
            return
        # Block NEW ghost attributes that shadow controller_parameters.simulator.
        # Attributes already in __dict__ (e.g. self.env) are legitimate instance
        # attributes — allow updating them.
        cp = self.__dict__.get('controller_parameters')
        if (cp is not None and hasattr(cp, 'simulator')
                and hasattr(cp.simulator, name) and name not in self.__dict__):
            raise AttributeError(
                f"Cannot set '{name}' directly on Simulator. "
                f"Use self.controller_parameters.simulator.{name} instead."
            )
        super().__setattr__(name, value)

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

        return cls(env=hydra.utils.get_class(simulator_config.env._target_).from_config(simulator_config.env),
                   controller_parameters=cp)
    
    def to_config(self, state):
        
        config_path = os.path.join(get_config_dir(), 'scene', 'simulator')
        with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = hydra.compose(config_name="base_simulator")
            cfg = OmegaConf.merge(
                cfg, 
                OmegaConf.create(
                    {'client': {
                        'controller_kwargs': {
                            'subtype_labels': self.controller_parameters.simulator.subtype_labels,
                            'freq': self.controller_parameters.simulator.freq,
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

    def _step(self, state):
        """Execute a step of the simulation.

        :param state: current simulation state
        :return: new state
        """
        return self.env.step(state=state)

    def step(self, changes=None):
        
        if changes is not None and len(changes) > 0:
            self.set_changes(changes)
        self.state = self._step(self.state)

        if changes is None:
            return self.state
        else:
            return StateAndControllerParameters(state=self.state, controller_parameters=self.controller_parameters)

    def run(self, threaded=False, num_steps=math.inf):
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
            _run = partial(self._run, num_steps=num_steps)
            threading.Thread(target=_run).start()
        else:
            self._run(num_steps=num_steps)

    def _run(self, num_steps):
        """Function that runs the simulator for the desired number of steps. Can be called either normally or in a thread.

        :param num_steps: number of simulation steps
        """
        lg.debug("Starting simulator _run")
        self._is_running = True
        lg.info("Simulation run starts")

        loop_count = 0

        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            self.sleep_timer.frequency = self.controller_parameters.simulator.freq
            with sleep_timer(timer=self.sleep_timer):

                if self._to_stop:
                    lg.debug("Stopping simulator _run as requested")
                    self._to_stop = False
                    break

                self.step()
                loop_count += 1

        self._is_running = False
        lg.info("Simulation run stops")

    def is_running(self):
        return self._is_running or self._was_running

    def set_changes(self, changes, update_from_server=True):
        # update_from_server is only here to match the SimulatorGRPCClient interface

        lg.debug("Applying changes to simulator")
        lg.debug(f"Changes: {changes}")
        update_dataclass_from_change_list(self, changes)

        # Sync sleep timer from source of truth
        self.sleep_timer.frequency = self.controller_parameters.simulator.freq

        # Sync env config values (box_size, num_scan_steps, …) to the actual Environment
        sync_dataclass_fields(self.env, self.controller_parameters.simulator.env)

        # Handle start/stop orchestration
        sim_params = self.controller_parameters.simulator
        if sim_params.run_from == self.name:
            if self.is_running() != sim_params.simulation_running:
                if sim_params.simulation_running:
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
            lg.debug(f"Resuming simulator: was_running={self._was_running}, simulation_running={self.controller_parameters.simulator.simulation_running}, _is_running={self._is_running}")
            if self._was_running and self.controller_parameters.simulator.simulation_running and not self._is_running:
                lg.debug("Running simulator from pause context manager")
                self.run(threaded=True)
            self._was_running = False
