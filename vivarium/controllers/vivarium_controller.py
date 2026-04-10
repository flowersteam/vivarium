import os
import math
import hydra
import logging
import threading
from time import sleep

from vivarium.runtime import (
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
from vivarium.controllers.handlers import Logger, RoutineHandler


logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)


class VivariumController:

    def __init__(self,
                 client=None,
                 connect_to_server=False,
                 start_server=False,
                 scene_name=None,
                 timeout=30.0,
                 start_controller_thread=True):
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
            start_controller_thread: Whether to start the controller thread immediately
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
        self._controller_thread = None
        self._controller_thread_stop_event = threading.Event()
        self._server_process = None
        self._interface_process = None
        self._ngrok_active = False
        self.interface_url = None
        self.logger = Logger()
        self.routine_handler = RoutineHandler()

        # Initialize in disconnected state
        self.client = None
        self.controllers = {}

        if client is not None:
            # Use provided client directly
            self._initialize_from_client(client, start_controller_thread=start_controller_thread)
        elif start_server:
            # Start server and connect
            if scene_name is None:
                raise ValueError("scene_name is required when start_server=True")
            self.start_server_process(scene_name, timeout=timeout, start_controller_thread=start_controller_thread)
        elif connect_to_server:
            # Try to connect to existing server
            self.connect(start_controller_thread=start_controller_thread)

    @classmethod
    def start_session(cls, scene_name,
                      client=None,
                      start_interface=False,
                      start_controller_thread=True,
                      step_from_controller=True,
                      run_simulation=True,
                      server_timeout=30.0):
        """Start a Vivarium session with server, simulation, and optionally interface.

        Args:
            scene_name: Name of the scene configuration to load
            client: Existing SimulatorGRPCClient or Simulator, or None to start a new server
            start_interface: Whether to start the Panel web interface
            start_controller_thread: Whether to start the controller thread immediately
            step_from_controller: Whether this controller drives simulation steps
            run_simulation: Whether to start the simulation running immediately
            server_timeout: Maximum seconds to wait for gRPC server to be ready

        Returns:
            VivariumController instance with interface_url attribute set
        """
        # Create controller - either with provided client or in disconnected state
        if client is not None:
            controller = cls(client=client, start_controller_thread=False)
        else:
            controller = cls()  # disconnected state
            controller.start_server_process(scene_name, timeout=server_timeout, start_controller_thread=start_controller_thread)
            if start_interface:
                controller.start_interface()

        if step_from_controller:
            controller.simulator.run_from = controller.client.name
        if start_controller_thread:
            controller.start_controller_thread()
        if run_simulation:
            controller.simulator.simulation_running = True
        controller.apply_changes()
        lg.info(f"VivariumController session '{scene_name}' is started")

        # Print the URL the user should use
        if controller.interface_url:
            print(f"\n🌐 Open the interface at: {controller.interface_url}\n")

        return controller

    def __getattr__(self, name):
        if name in self.controllers:
            return self.controllers[name]
        raise AttributeError(f"'VivariumController' object has no attribute '{name}'")

    def _initialize_from_client(self, client, scene_config=None, start_controller_thread=True):
        """Initialize controllers from an existing client.

        Args:
            client: SimulatorGRPCClient or Simulator instance.
            scene_config: Optional scene configuration. If not provided,
                         will be loaded based on client.scene_name.
        """
        self.client = client
        scene_config = scene_config or load_scene_config(client.scene_name)
        components_config = scene_config.environment.components

        # Fetch subtype labels once. This list object is shared by reference across
        # all entity/consumption/spawn controllers constructed below, so that
        # set_subtype_labels() can update them all via a single in-place mutation.
        subtype_labels = client.remote.controller_parameters.simulator.subtype_labels.obj()

        # Load controllers from scene config
        controllers = {}
        for name, c_config in components_config.component_list.items():
            if 'client' in c_config and 'controller_cls' in c_config.client:
                c_cls = hydra.utils.get_class(c_config.client.controller_cls)
                controllers[name] = c_cls.from_config(name, client.remote)

        self.controllers = controllers
        self.controllers['simulator'] = SimulatorController(name='simulator', remote=self.client.remote)
        self.subtypes = subtype_labels  # same object as in all controllers above
        if start_controller_thread:
            self.start_controller_thread()

    def set_subtype_labels(self, new_labels):
        """Rename subtype labels on this client.

        Updates the label list in-place, so the change is immediately visible in
        all entity, consumption, and spawn controllers (they all share the same
        underlying list object).

        You may provide fewer labels than the total number of subtypes; the
        remaining ones keep their current labels. len(new_labels) must not
        exceed len(self.subtypes).

        Args:
            new_labels: list of new label strings, at most len(self.subtypes) long.
        """
        self.ensure_connected()
        if len(new_labels) > len(self.subtypes):
            raise ValueError(
                f"Expected at most {len(self.subtypes)} labels, got {len(new_labels)}. "
                f"Current subtypes: {self.subtypes}"
            )
        self.subtypes[:] = list(new_labels) + self.subtypes[len(new_labels):]

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

    def connect(self, reconnect=False, start_controller_thread=True):
        """Connect to an existing gRPC server.

        Args:
            timeout: Timeout in seconds for connection attempt.
            reconnect: If True, clean up any existing (potentially stale) connection
                      and establish a fresh one. Useful when the server was restarted
                      by another client.
            start_controller_thread: Whether to start the controller thread immediately
                                   after connection (default: True).

        Returns:
            True if connected successfully, False otherwise.
        """
        if reconnect and self.client is not None:
            # Clean up potentially stale connection
            if check_server_running():
                lg.info("Reconnecting to server...")
                try:
                    self.stop_controller_thread()
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
            self._initialize_from_client(client, start_controller_thread=start_controller_thread)
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

        if self.is_controller_thread_running():
            self.stop_controller_thread()
            if self._controller_thread is not threading.current_thread():
                self._controller_thread.join(timeout=2.0)
        # Close the client connection
        try:
            self.client.close()
        except Exception as e:
            lg.warning(f"Error closing client: {e}")
        self.client = None
        self.controllers = {}
        lg.info("Disconnected from server")

    def start_server_process(self, scene_name, timeout=30.0, start_controller_thread=True):
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
                self._initialize_from_client(client, start_controller_thread=start_controller_thread)
            else:
                lg.info(f"Already connected to server with scene '{scene_name}'.")
            return

        # Start the server
        self._server_process = start_simulation_server(scene_name, timeout=timeout)
        lg.info(f"Server started for scene '{scene_name}'")

        # Connect to it
        client = SimulatorGRPCClient()
        self._initialize_from_client(client, start_controller_thread=start_controller_thread)

    def stop_server_process(self):
        """Stop the server process if we started it.

        This also disconnects the controller since the server no longer exists.
        Called automatically by close() if the controller started the server.
        Can also be called explicitly if you want to stop the server.
        """
        if self._server_process is not None:
            # Disconnect first to stop threads cleanly before server goes down
            self.disconnect()
            # Now stop the server
            stop_simulation_server(self._server_process)
            self._server_process = None
            lg.info("Server stopped")

    def start_controller_thread(self, threaded=True, num_steps=math.inf, debug_mode=False):
        """
        Start the controller thread to maintain synchronization with the simulator server.
        The simulator step will be called from this controller if `self.simulator.run_from == self.client.name` (synchronous mode).
        :param threaded: Whether to run in a separate thread or not, defaults to True
        :raises RuntimeError: if the simulator is already started
        """

        self.ensure_connected()

        if self.is_controller_thread_running():
            lg.info("Controller thread is already running")
            return

        # automatically catch errors only if not in debug mode
        catch_errors = not debug_mode

        self._controller_thread_stop_event.clear()
        if threaded:
            self._controller_thread = threading.Thread(
                target=self._run_controller_thread, args=(num_steps, catch_errors)
            )
            self._controller_thread.daemon = True
            self._controller_thread.start()
        else:
            self._run_controller_thread(num_steps=num_steps, catch_errors=catch_errors)
        lg.info("Controller thread started on client")

    def _run_controller_thread(self, num_steps=math.inf, catch_errors=True):
        """Run the simulation for a given number of steps.

        :param num_steps: num_steps, defaults to math.inf
        :param catch_errors: whether to catch errors or not, defaults to False
        """
        run_time = 0
        while run_time < num_steps and not self._controller_thread_stop_event.is_set():
            with sleep_timer(freq=self.controllers['simulator'].freq):
                self.step(catch_errors=catch_errors)
                run_time += 1

        # finally stop the simulation
        if self.is_controller_thread_running():
            self.stop_controller_thread()


    def stop_controller_thread(self):
        """Stop controller thread on this client."""
        if not self.is_controller_thread_running():
            lg.info("Controller thread is already stopped")
        self._controller_thread_stop_event.set()

    def is_controller_thread_running(self):
        """Check if the controller thread is running on this client."""
        return self._controller_thread is not None and self._controller_thread.is_alive()

    def simulator_step(self):
        changes = self.fetch_changes()
        self.client.step(changes)

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the controller.

        :param routine_fn: routine function that takes the controller as argument
        :param name: routine name, defaults to None (routine function name)
        :param interval: routine execution interval, defaults to 1
        """
        self.routine_handler.attach_routine(routine_fn, name, interval)

    def detach_routine(self, routine_fn):
        """Detach a routine from the controller.

        :param routine_fn: routine function or its name as a string
        """
        self.routine_handler.detach_routine(routine_fn)

    def detach_all_routines(self):
        """Detach all routines from the controller."""
        self.routine_handler.detach_all_routines()
        
    def print_routines(self):
        """Print the controller routines"""
        self.routine_handler.print_routines()        

    def controller_step(self, catch_errors=True):
        # Step through controllers (e.g. routines and behaviors)
        for _, controller in self.controllers.items():
            controller.step(time=self.time, catch_errors=catch_errors)
        self.routine_handler.routine_step(self, self.time, catch_errors)

    def step(self, catch_errors=True):
        changed_applied = False
        if self.simulator.simulation_running:
            self.controller_step(catch_errors=catch_errors)
            self.time += 1
            if self.simulator.run_from == self.client.name: # and self.simulator.simulation_running:
                self.simulator_step()
                changed_applied = True
        if not changed_applied:
            self.apply_changes()

    def fetch_changes(self):
        return self.client.remote.fetch_changes()

    def apply_changes(self, changes=None, close_if_requested=True):
        changes = changes or self.fetch_changes()
        # Use set_changes when streaming is active to avoid redundant state fetch
        update_from_server = not (hasattr(self.client, 'is_streaming') and self.client.is_streaming)
        self.client.set_changes(changes, update_from_server=update_from_server)
        if self.simulator.close and close_if_requested:
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
                from vivarium.runtime import create_ngrok_tunnel
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
                from vivarium.runtime import close_ngrok_tunnel
                close_ngrok_tunnel()
                self._ngrok_active = False            
            
            lg.info("Interface stopped")

    def close(self):
        """Close the controller and clean up all resources.

        Stops any processes we started (interface, server) and disconnects from the server.
        """
        # Stop interface if we started it
        if self._interface_process is not None:
            self.stop_interface()

        # Stop server if we started it (this also calls disconnect())
        if self._server_process is not None:
            self.stop_server_process()
        else:
            # Just disconnect if we didn't start the server
            self.disconnect()

    def close_all(self):
        """Send signal to close all clients and the simulator."""
        self.disconnect()
        if 'simulator' in self.controllers:
            self.simulator.close = True
            lg.info("Waiting for all clients to close ...")
            while len(self.simulator.client_names) != 1:  # wait for other clients to close
                if self.is_connected(verify=True):
                    self.apply_changes(close_if_requested=False)
                else:
                    break
                sleep(0.1)
        # Close our client
        self.close()

    def close_session(self, safe_mode=False):
        """Stop the session: simulation, server, interface, and ngrok tunnel"""
        self.close_all()
        stop_server_and_interface(safe_mode=safe_mode)
