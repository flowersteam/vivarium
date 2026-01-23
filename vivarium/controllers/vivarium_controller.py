import math
import hydra
import logging
import threading
from time import sleep

from vivarium.utils.handle_server_interface import (
    start_server_and_interface,
    stop_server_and_interface,
    create_ngrok_tunnel,
    start_simulation_server,
    stop_simulation_server,
    check_server_running,
)
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.controller import SimulatorController
from vivarium.utils.scene_configs import load_scene_config
from vivarium.utils.timer import sleep_timer


logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)


class VivariumController:

    def __init__(self, client=None, subtypes=[], server_process=None, **controllers):
        self.client = client or SimulatorGRPCClient()
        self.subtypes = subtypes

        self.controllers = controllers

        self.time = 0
        self._is_started = False

        # Track server process if we started it (for cleanup on close)
        self._server_process = server_process

        self.controllers['simulator'] = SimulatorController(name='simulator', remote=self.client.remote)

    @classmethod
    def from_client(cls, client=None, scene_config=None, server_process=None):
        client = client or SimulatorGRPCClient()
        scene_config = scene_config or load_scene_config(client.scene_name)
        components_config = scene_config.environment.components
        controllers = {}
        for name, c_config in components_config.component_list.items():
            if 'client' in c_config and 'controller_cls' in c_config.client:
                c_cls = hydra.utils.get_class(c_config.client.controller_cls)
                controllers[name] = c_cls.from_config(name, client.remote)
        return cls(
            client=client,
            subtypes=components_config.subtype_labels,
            server_process=server_process,
            **controllers
        )

    @classmethod
    def start_server(cls, scene_name, timeout=30.0):
        """Start a simulation server and return a controller connected to it.

        This is a convenience method for starting a server without the web interface.
        Useful for Jupyter notebooks or programmatic control.

        Args:
            scene_name: Name of the scene configuration to load (e.g., 'quickstart', 'session_1')
            timeout: Maximum seconds to wait for server to be ready

        Returns:
            VivariumController instance connected to the server.
            The controller tracks the server process and will stop it when close() is called.

        Example:
            controller = VivariumController.start_server('quickstart')
            controller.simulator.simulation_running = True
            # ... interact with the simulation ...
            controller.close()  # This also stops the server
        """
        # Check if server is already running
        if check_server_running():
            lg.warning("Server is already running. Connecting to existing server.")
            return cls.from_client()

        # Start the server
        server_process = start_simulation_server(scene_name, timeout=timeout)
        lg.info(f"Server started for scene '{scene_name}'")

        # Connect and create controller
        client = SimulatorGRPCClient()
        controller = cls.from_client(client=client, server_process=server_process)

        return controller

    @classmethod
    def start_session(cls, scene_name,
                      client=None,
                      start_interface=True,
                      safe_mode=False,
                      step_from_controller=True, 
                      run_simulation=True,
                      start=True,
                      server_timeout=30.0,
                      ngrok=False,
                      ngrok_token=None):
        """Start a Vivarium session with server, simulation, and optionally interface.
        
        Args:
            scene_name: Name of the scene configuration to load
            client: Existing SimulatorGRPCClient or Simulator, or None to start a new server
            start_interface: Whether to start the Panel web interface
            safe_mode: Whether to prompt before stopping existing processes
            step_from_controller: Whether this controller drives simulation steps
            run_simulation: Whether to start the simulation running immediately
            server_timeout: Maximum seconds to wait for gRPC server to be ready
            ngrok: Whether to create an ngrok tunnel for public access
            ngrok_token: ngrok auth token (reads from NGROK_TOKEN env var if None)
            
        Returns:
            VivariumController instance with interface_url attribute set
        """
        interface_url = None
        if client is None:
            interface_url = start_server_and_interface(cmd_args=[f'scene={scene_name}'], 
                                    start_interface=start_interface,
                                    safe_mode=safe_mode,
                                    server_timeout=server_timeout,
                                    allow_external_origins=ngrok)
        controller = cls.from_client(client=client)
        controller.interface_url = interface_url
        controller._ngrok_active = False
        
        # Create ngrok tunnel if requested
        if ngrok:
            try:
                ngrok_url = create_ngrok_tunnel(port=5006, token=ngrok_token)
                controller.interface_url = ngrok_url
                controller._ngrok_active = True
            except Exception as e:
                lg.warning(f"Failed to create ngrok tunnel: {e}")
        
        if step_from_controller:
            controller.simulator.run_from = controller.client.name
        if start:
            controller.start()
        if run_simulation:
            controller.simulator.simulation_running = True
        lg.info(f"VivariumController session '{scene_name}' is started")
        
        # Print the URL the user should use
        if controller.interface_url:
            print(f"\n🌐 Open the interface at: {controller.interface_url}\n")
        
        return controller                 

    def __getattr__(self, name):
        if name in self.controllers:
            return self.controllers[name]
        raise AttributeError(f"'VivariumController' object has no attribute '{name}'")

    def start(self, threaded=True, num_steps=math.inf, debug_mode=False, use_streaming=False):
        """
        Execute the simulation loop from this client.
        :param threaded: Whether to run the simulation in a thread or not, defaults to True
        :param use_streaming: Use bidirectional streaming (True) or unary RPC (False), defaults to True
        :raises RuntimeError: if the simulator is already started
        """
        if self.is_started():
            lg.info("Simulator is already started")
            return

        # automatically catch errors only if not in debug mode
        catch_errors = not debug_mode
        
        self._is_started = True
        if threaded:
            run_thread = threading.Thread(
                target=self._start, args=(num_steps, catch_errors, use_streaming)
            )
            run_thread.daemon = True
            run_thread.start()
        else:
            self._start(num_steps=num_steps, catch_errors=catch_errors, use_streaming=use_streaming)
        lg.info("Simulator started on client")
            
    def _start(self, num_steps=math.inf, catch_errors=True, use_streaming=False):
        """Run the simulation for a given number of steps.

        :param num_steps: num_steps, defaults to math.inf
        :param catch_errors: wether to catch errors or not, defaults to False
        :param use_streaming: Use bidirectional streaming (True) or unary RPC (False)
        """
        
        if use_streaming:
            # Use synchronized bidirectional streaming
            def compute_changes():
                """Compute motor commands and return changes for the next step."""
                if not self._is_started:
                    return None  # Signal to stop
                
                with sleep_timer(freq=self.controllers['simulator'].freq):
                    self.controller_step(catch_errors=catch_errors)
                    self.time += 1
                    
                return self.fetch_changes()
            
            for state_and_cp in self.client.bidirectional_step_sync(num_steps, compute_changes):
                if not self._is_started:
                    break  # Stop if we've been told to stop
        else:
            # Use unary RPC (original approach)
            run_time = 0
            while run_time < num_steps and self._is_started:
                with sleep_timer(freq=self.controllers['simulator'].freq):
                    self.step(catch_errors=catch_errors)
                    self.time += 1
                    run_time += 1
                
        # finally stop the simulation
        if self.is_started():
            self.stop()


    def stop(self):
        """Stop simulation loop on this client."""
        if not self.is_started():
            lg.info("Simulator is already stopped")
        self._is_started = False

    def is_started(self):
        """Check if the simulation loop is started on this client."""
        return self._is_started

    def simulator_step(self):
        changes = self.fetch_changes()
        self.client.step(changes)
        
    def controller_step(self, catch_errors=True):
        # Step through controllers (e.g. routines and behaviors)
        for _, controller in self.controllers.items():
            controller.step(time=self.time, catch_errors=catch_errors)        

    def step(self, catch_errors=True):
        changed_applied = False
        if self.simulator.simulation_running:
            self.controller_step(catch_errors=catch_errors)
            if self.simulator.run_from == self.client.name: # and self.simulator.simulation_running:
                self.simulator_step()
                changed_applied = True
        if not changed_applied:
            self.apply_changes()

    def fetch_changes(self):
        return self.client.remote.fetch_changes()

    def apply_changes(self, changes=None, close_if_resquested=True): # TODO: should this be in SimulatorClient instead?
        changes = changes or self.fetch_changes()
        # Use set_changes when streaming is active to avoid redundant state fetch
        update_from_server = not (hasattr(self.client, 'is_streaming') and self.client.is_streaming)
        self.client.set_changes(changes, update_from_server=update_from_server)
        if self.simulator.close:
            self.close() 
        
    def close(self):
        if self._is_started:
            self.stop()
            sleep(4)  # wait for the simulation loop to stop
        self.client.close()

        # Close ngrok tunnel if one was created
        if getattr(self, '_ngrok_active', False):
            from vivarium.utils.handle_server_interface import close_ngrok_tunnel
            close_ngrok_tunnel()
            self._ngrok_active = False

        # Stop server if we started it
        if self._server_process is not None:
            self.stop_server()

    def stop_server(self):
        """Stop the server if this controller started it.

        This is called automatically by close() if the controller started the server.
        Can also be called explicitly if you want to stop the server without closing the controller.
        """
        if self._server_process is not None:
            stop_simulation_server(self._server_process)
            self._server_process = None
            lg.info("Server stopped")
            
    def close_all(self):
        """Send signal to close all clients and the simulator."""
        self.stop()
        sleep(1)  # wait for the simulation loop to stop
        self.simulator.close = True
        lg.info("Waiting for all clients to close ...")
        while len(self.simulator.client_names) != 1:  # wait for other clients to close
            self.apply_changes(close_if_resquested=False)
            sleep(0.1)
        # Close our client
        self.close()
            
    def close_session(self, safe_mode=False):
        """Stop the session: simulation, server, interface, and ngrok tunnel"""
        self.close_all()
        stop_server_and_interface(safe_mode=safe_mode)
        