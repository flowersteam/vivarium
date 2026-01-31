import os
import math
import hydra
import logging
import threading
from time import sleep

from vivarium.utils.handle_server_interface import (
    start_simulation_server,
    stop_simulation_server,
    check_server_running,
    start_panel_interface,
    stop_panel_interface,
    stop_server_and_interface,
)
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.controller import SimulatorController
from vivarium.utils.scene_configs import load_scene_config
from vivarium.utils.timer import sleep_timer


logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)


class VivariumController:

    def __init__(self,
                 client=None,
                 connect_to_server=False,
                 start_server=False,
                 scene_name=None,
                 timeout=30.0,
                 start_controller_loop=True):
        """Initialize a VivariumController.

        Args:
            client: Existing SimulatorGRPCClient or Simulator to use. If provided,
                    the controller will initialize immediately using this client.
            connect_to_server: If True and no client provided, attempt to connect
                              to an existing gRPC server.
            start_server: If True and no client provided, start a new server.
                         Requires scene_name to be set.
            scene_name: Name of the scene configuration (required if start_server=True).
            timeout: Timeout in seconds for server connection/startup.
            start_controller_loop: Whether to start the controller loop immediately
                                   after initialization (default: True).

        Behavior matrix:
            | client | connect_to_server | start_server | Behavior |
            |--------|-------------------|--------------|----------|
            | Provided | - | - | Use client directly, initialize controllers |
            | None | False | False | Disconnected state - no client |
            | None | True | False | Connect if server running, else warning |
            | None | False | True | Start server (requires scene_name), then connect |
            | None | True | True | If server running: connect. If not: start, then connect |
        """
        self.time = 0
        self._is_started = False
        self._server_process = None
        self._interface_process = None
        self._ngrok_active = False
        self.interface_url = None

        # Initialize in disconnected state
        self.client = None
        self.controllers = {}

        if client is not None:
            # Use provided client directly
            self._initialize_from_client(client, start_controller_loop=start_controller_loop)
        elif start_server:
            # Start server and connect
            if scene_name is None:
                raise ValueError("scene_name is required when start_server=True")
            self.start_server_process(scene_name, timeout=timeout, start_controller_loop=start_controller_loop)
        elif connect_to_server:
            # Try to connect to existing server
            self.connect(timeout=timeout)

    @classmethod
    def start_session(cls, scene_name,
                      client=None,
                      start_interface=True,
                      start_controller_loop=True,
                      step_from_controller=True,
                      run_simulation=True,
                      server_timeout=30.0):
        """Start a Vivarium session with server, simulation, and optionally interface.

        Args:
            scene_name: Name of the scene configuration to load
            client: Existing SimulatorGRPCClient or Simulator, or None to start a new server
            start_interface: Whether to start the Panel web interface
            start_controller_loop: Whether to start the controller loop immediately
            step_from_controller: Whether this controller drives simulation steps
            run_simulation: Whether to start the simulation running immediately
            server_timeout: Maximum seconds to wait for gRPC server to be ready

        Returns:
            VivariumController instance with interface_url attribute set
        """
        # Create controller - either with provided client or in disconnected state
        if client is not None:
            controller = cls(client=client)
        else:
            controller = cls()  # disconnected state
            controller.start_server_process(scene_name, timeout=server_timeout)
            if start_interface:
                controller.start_interface()

        if step_from_controller:
            controller.simulator.run_from = controller.client.name
        if start_controller_loop:
            controller.start_controller_loop()
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

    def _initialize_from_client(self, client, scene_config=None, start_controller_loop=True):
        """Initialize controllers from an existing client.

        Args:
            client: SimulatorGRPCClient or Simulator instance.
            scene_config: Optional scene configuration. If not provided,
                         will be loaded based on client.scene_name.
        """
        self.client = client
        scene_config = scene_config or load_scene_config(client.scene_name)
        components_config = scene_config.environment.components

        # Load controllers from scene config
        controllers = {}
        for name, c_config in components_config.component_list.items():
            if 'client' in c_config and 'controller_cls' in c_config.client:
                c_cls = hydra.utils.get_class(c_config.client.controller_cls)
                controllers[name] = c_cls.from_config(name, client.remote)

        self.controllers = controllers
        self.controllers['simulator'] = SimulatorController(name='simulator', remote=self.client.remote)
        if start_controller_loop:
            self.start_controller_loop()

    def is_connected(self, verify=True):
        """Check if connected to a server.

        Args:
            verify: If True (default), actively ping the server to verify it's still
                   responding. If False, only check local state.

        Returns:
            True if connected to a server, False otherwise.
        """
        if self.client is None:
            return False
        else:
            if not self.client.is_grpc_client:
                lg.warning("Client is not a gRPC client; assuming connected.")
                return True  # gRPC client assumed to be connected if client exists

        if verify:
            # Actually check if server is still responding
            host = self.client.server_host
            port = self.client.server_port
            if not check_server_running(host=host, port=port):
                # Server is down, clean up local state
                lg.warning("Server is no longer responding. Marking as disconnected.")
                self.client = None
                self.controllers = {}
                return False

        return True

    def ensure_connected(self, verify=False):
        """Guard method - raises RuntimeError if not connected."""
        if not self.is_connected(verify=verify):
            raise RuntimeError(
                "Not connected to a server. Use connect(), start_server_process(), "
                "or pass a client/start_server=True to the constructor."
            )

    def connect(self, timeout=30.0, reconnect=False, start_controller_loop=True):
        """Connect to an existing gRPC server.

        Args:
            timeout: Timeout in seconds for connection attempt.
            reconnect: If True, clean up any existing (potentially stale) connection
                      and establish a fresh one. Useful when the server was restarted
                      by another client.

        Returns:
            True if connected successfully, False otherwise.
        """
        if reconnect and self.client is not None:
            # Clean up potentially stale connection
            if check_server_running():
                lg.info("Reconnecting to server...")
                try:
                    self.stop_controller_loop()
                    self.client.close()
                except Exception:
                    pass  # Ignore errors from stale connection
                self.client = None
                self.controllers = {}
            else:
                lg.warning("No server running. Use start_server_process() to start one.")
                return False
        elif self.is_connected():
            lg.warning("Already connected to a server. Use reconnect=True to force reconnection.")
            return True

        if not check_server_running():
            lg.warning("No server running. Use start_server_process() to start one.")
            return False

        try:
            client = SimulatorGRPCClient()
            self._initialize_from_client(client, start_controller_loop=start_controller_loop)
            lg.info("Connected to gRPC server")
            return True
        except Exception as e:
            lg.error(f"Failed to connect to server: {e}")
            return False

    def disconnect(self):
        """Disconnect from the server and clean up controllers."""
        if not self.is_connected():
            lg.info("Already disconnected")
            return

        self.stop_controller_loop()
        sleep(1)  # wait for the simulation loop to stop
        # Close the client connection
        try:
            self.client.close()
        except Exception as e:
            lg.warning(f"Error closing client: {e}")
        self.client = None
        self.controllers = {}
        lg.info("Disconnected from server")

    def start_server_process(self, scene_name, timeout=30.0, start_controller_loop=True):
        """Start a server process and connect to it.

        Args:
            scene_name: Name of the scene configuration to load.
            timeout: Timeout in seconds for server startup.

        Raises:
            RuntimeError: If server fails to start within timeout.
        """
        # Check if server is already running
        if check_server_running():
            # Check which scene is running (use existing client if connected)
            if self.client is not None:
                running_scene = self.client.scene_name
            else:
                client = SimulatorGRPCClient()
                running_scene = client.scene_name

            if running_scene != scene_name:
                if self.client is None:
                    client.close()
                lg.warning(
                    f"Server is already running with scene '{running_scene}' (requested: '{scene_name}'). "
                    f"Use connect() to connect to it, or stop_server_and_interface() to stop it first."
                )
                return

            # Same scene - connect if not already connected
            if self.client is None:
                lg.info(f"Server already running with scene '{scene_name}'. Connecting.")
                self._initialize_from_client(client, start_controller_loop=start_controller_loop)
            else:
                lg.info(f"Already connected to server with scene '{scene_name}'.")
            return

        # Start the server
        self._server_process = start_simulation_server(scene_name, timeout=timeout)
        lg.info(f"Server started for scene '{scene_name}'")

        # Connect to it
        client = SimulatorGRPCClient()
        self._initialize_from_client(client, start_controller_loop=start_controller_loop)

    def stop_server_process(self):
        """Stop the server process if we started it.

        This also disconnects the controller since the server no longer exists.
        Called automatically by close() if the controller started the server.
        Can also be called explicitly if you want to stop the server.
        """
        if self._server_process is not None:
            stop_simulation_server(self._server_process)
            self._server_process = None
            lg.info("Server stopped")
            # Also disconnect since the server we were connected to no longer exists
            self.disconnect()

    def start_controller_loop(self, threaded=True, num_steps=math.inf, debug_mode=False, use_streaming=False):
        """
        Execute the controller loop to maintain synchronization with the simulator server.
        The simulator step will be called from this controller if `self.simulator.run_from == self.client.name` (synchronous mode).
        :param threaded: Whether to run the loop in a thread or not, defaults to True
        :param use_streaming: Use bidirectional streaming (True) or unary RPC (False), defaults to True
        :raises RuntimeError: if the simulator is already started
        """
        
        self.ensure_connected()

        if self.is_started():
            lg.info("Simulator is already started")
            return

        # automatically catch errors only if not in debug mode
        catch_errors = not debug_mode

        self._is_started = True
        if threaded:
            run_thread = threading.Thread(
                target=self._start_controller_loop, args=(num_steps, catch_errors, use_streaming)
            )
            run_thread.daemon = True
            run_thread.start()
        else:
            self._start_controller_loop(num_steps=num_steps, catch_errors=catch_errors, use_streaming=use_streaming)
        lg.info("Simulator started on client")

    def _start_controller_loop(self, num_steps=math.inf, catch_errors=True, use_streaming=False):
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
            self.stop_controller_loop()


    def stop_controller_loop(self):
        """Stop controller loop on this client."""
        if not self.is_started():
            lg.info("Controller loop is already stopped")
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
            
    def start_interface(self, show_output=True, timeout=10, use_ngrok=False, ngrok_token=None):
        """Start the Panel web interface.

        Args:
            show_output: Whether to show interface output in console.
            timeout: Timeout in seconds for interface startup.
            use_ngrok: Whether to create an ngrok tunnel for public access.
            ngrok_token: ngrok auth token (reads from NGROK_TOKEN env var if None).

        Returns:
            Interface URL if started successfully, None otherwise.
        """

        if self._interface_process is not None:
            lg.warning("Interface is already running")
            return self.interface_url
        
        if os.environ.get('VIVARIUM_JUPYTER_FROM_PANEL') == '1':
            lg.info("Jupyter was started from Panel interface; skipping interface start. "
                    "If you want to start another interface, set the environment variable VIVARIUM_JUPYTER_FROM_PANEL to 0.")
            return self.interface_url

        self._interface_process, self.interface_url = start_panel_interface(
            allow_external_origins=use_ngrok,
            show_output=show_output,
            timeout=timeout
        )
        
        # Create ngrok tunnel if requested
        if use_ngrok:
            try:
                from vivarium.utils.handle_server_interface import create_ngrok_tunnel
                ngrok_url = create_ngrok_tunnel(port=5006, token=ngrok_token)
                self.interface_url = ngrok_url
                self._ngrok_active = True
            except Exception as e:
                lg.warning(f"Failed to create ngrok tunnel: {e}")        

        if self.interface_url:
            lg.info(f"Interface started at: {self.interface_url}")
        return self.interface_url

    def stop_interface(self):
        """Stop the Panel interface if we started it."""
        if self._interface_process is not None:
            stop_panel_interface(self._interface_process)
            self._interface_process = None
            self.interface_url = None
            
            # Close ngrok tunnel if one was created
            if getattr(self, '_ngrok_active', False):
                from vivarium.utils.handle_server_interface import close_ngrok_tunnel
                close_ngrok_tunnel()
                self._ngrok_active = False            
            
            lg.info("Interface stopped")

    def close(self):
        if self._is_started:
            self.stop_controller_loop()
            sleep(1)  # wait for the simulation loop to stop

        # Close client if connected
        if self.is_connected():
            self.client.close()
            self.client = None
            self.controllers = {}

        # Stop interface if we started it
        if self._interface_process is not None:
            self.stop_interface()

        # Stop server if we started it
        if self._server_process is not None:
            self.stop_server_process()

    def close_all(self):
        """Send signal to close all clients and the simulator."""
        self.stop_controller_loop()
        sleep(1)  # wait for the simulation loop to stop
        if 'simulator' in self.controllers:
            self.simulator.close = True
            lg.info("Waiting for all clients to close ...")
            while len(self.simulator.client_names) != 1:  # wait for other clients to close
                self.apply_changes(close_if_resquested=False)
                sleep(0.1)
        # close all processes
        stop_server_and_interface(safe_mode=False)
        # Close our client
        self.close()

    def close_session(self, safe_mode=False):
        """Stop the session: simulation, server, interface, and ngrok tunnel"""
        self.close_all()
        stop_server_and_interface(safe_mode=safe_mode)
