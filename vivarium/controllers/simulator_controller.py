import math
import logging
import hydra
import threading
from dataclasses import asdict, fields

from vivarium.utils.handle_server_interface import start_server_and_interface, stop_server_and_interface
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import SimulatorParametersWrapper

from vivarium.utils.scene_configs import load_scene_config


logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)


def start_session(scene_name, run=True):
    start_server_and_interface(cmd_args=[f'scene={scene_name}'])
    scene_config = load_scene_config(scene_name)
    components_config = scene_config.environment.components
    controller = SimulatorController.from_config(config=components_config, run_from_server=scene_config.simulator.run_from_server)
    if run:
        controller.run()
    return controller

class SimulatorController:

    def __init__(self, client=None, run_from_server=True, subtypes=[], **controllers):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.simulator_parameters = self.client.get_simulator_parameters()
        self.subtype_labels = {i: label for i, label in enumerate(subtypes)}
        
        self.controllers = controllers
        
        self.time = 0
        self.run_from_server = run_from_server
        self._is_running = False
        
        self.create_simulator_parameters_wrapper()

    @classmethod
    def from_config(cls, config, client=None, run_from_server=True):
        client = client or SimulatorGRPCClient()
        state = client.state
        cp = asdict(client.controller_parameters)
        controllers = {}
        for name, c_config in config.component_list.items():
            if 'client' in c_config and 'controller_cls' in c_config.client:
                c_cls = hydra.utils.get_class(c_config.client.controller_cls)
                p = {} if name not in cp or cp[name] is None else cp[name]
                controllers[name] = c_cls.from_config(name, c_config.client, state, notebook_control=not run_from_server, **p)
        return cls(
            client=client,
            run_from_server=run_from_server,
            subtypes=config.subtype_labels,
            **controllers
        )

    def __getattr__(self, name):
        if name in self.controllers:
            return self.controllers[name]
        raise AttributeError(f"'SimulatorController' object has no attribute '{name}'")

    def create_simulator_parameters_wrapper(self):
        self.simulator_parameters = SimulatorParametersWrapper(self.simulator_parameters)

    def run(self, threaded=True, num_steps=math.inf, debug_mode=False):
        """
        Run the simulation.
        :param threaded: if self.run_from_server is False, whether to run the simulation in a thread or not, defaults to True
        :param num_steps: if self.run_from_server is False, num_steps, defaults to math.inf
        :raises RuntimeError: if the simulator is already started
        """
        if self.is_running():
            lg.info("Simulator is already started")
            return
        if self.run_from_server:
            self.client.start()
        else:
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
        self._is_running = True
            
    def _run(self, num_steps=math.inf, catch_errors=True):
        """run the simulation for a given number of steps

        :param num_steps: num_steps, defaults to math.inf
        :param catch_errors: wether to catch errors or not, defaults to False
        """
        # Add a local time for the run function independant from the controller time
        run_time = 0
        while run_time < num_steps and self._is_running:
            # self.execute_routines_and_behaviors(catch_errors=catch_errors)

            self.step()

            self.time += 1
            run_time += 1

        # finally stop the simulation
        self.stop()


    def stop(self):
        """Stop the simulator."""
        if not self.is_running():
            lg.info("Simulator is already stopped")
        else:
            if self.run_from_server:
                self.client.stop()
        self._is_running = False

    def is_running(self):
        """Check if the simulator is started."""
        if self.run_from_server:
            return self.client.is_running()
        return self._is_running

    def step(self):
        changes = self.fetch_changes()
        state_and_cp = self.client.step(changes)
        self.state = state_and_cp.state
        self.update_controllers(state=self.state, controller_parameters=state_and_cp.controller_parameters)

    def update_controllers(self, state=None, controller_parameters=None):
        """Update the controllers."""
        if state is None and controller_parameters is None:
            lg.warning("No state or controller parameters provided to update controllers")
        if controller_parameters is not None:
            cp_fields = [f.name for f in fields(controller_parameters)]
        for name, controller in self.controllers.items():
            if state is not None:
                controller.set_state(state)
            if controller_parameters is not None:
                if name in cp_fields:
                    controller.set_controller_parameters(getattr(controller_parameters, name))

    def update_state(self):
        """Update the state from server to client."""
        self.state = self.client.get_state()
        self.update_controllers(state=self.state)
        return self.state

    def fetch_changes(self):
        changes = []
        for _, controller in self.controllers.items():
            change = controller.fetch_changes()
            changes.extend(change)

        change = self.simulator_parameters.fetch_changes()
        if change:
            changes.extend([change])
        return changes

    def apply_changes(self, changes=None):
        changes = changes or self.fetch_changes()
        if len(changes) > 0:
            cp = self.client.apply_changes(changes)
        else:
            cp = self.client.controller_parameters
        self.update_controllers(controller_parameters=cp)
            
    def stop_session(self, safe_mode=False):
        """Stop the session: simulation, server and interface"""
        if self._is_running:
            self.stop()
        stop_server_and_interface(safe_mode=safe_mode)