import os
import time
import hydra
import logging
import threading
import panel as pn
from param import Parameterized

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    PointDrawTool,
    HoverTool,
    Range1d,
)

from vivarium.controllers import VivariumController
from vivarium.utils.scene_configs import load_scene_config, get_available_scenes
from vivarium.utils.runtime import get_bundle_root
from vivarium.utils.handle_server_interface import (
    check_server_running,
    get_server_interface_pids,
    terminate_process,
    kill_vivarium_processes,
    check_jupyter_running,
    register_jupyter_port,
    unregister_jupyter_port,
    find_next_available_port,
)
from vivarium.interface.parameterized import ParamSimulator
from vivarium.interface.utils import cleanup_parameterized_class
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.utils.handle_server_interface import start_jupyter_server


lg = logging.getLogger(__name__)


def create_interfaces(component_list_config, controllers, state, panel_cls=pn.Column):
    cleanup_parameterized_class()
    interfaces = {}
    for name, component in component_list_config.items():
        if 'client' in component:
            if 'interface_cls' in component.client:
                interface_cls = hydra.utils.get_class(component.client.interface_cls)
                interfaces[name] = interface_cls(
                    controllers[name],
                    panel_cls=panel_cls
                )
    return interfaces


class WindowManager(Parameterized):

    def __init__(self, controller=None, apply_changes=True, notebook_mode=False, testing_mode=False, server_timeout=30.0, **kwargs):
        super().__init__(**kwargs)

        # Basic state
        self.apply_changes = apply_changes
        self.testing_mode = testing_mode
        self.server_timeout = server_timeout
        self._streaming_active = False
        self._pending_state_update = threading.Event()
        self._state_lock = threading.Lock()

        # Track whether we started the server (for UI logic, actual process is managed by controller)
        self._started_server = False

        # TODO: Obsolete, to remove here and all other modules using it
        self.notebook_mode = notebook_mode

        self.curdoc = curdoc()

        # Check theme from URL query params (before server connection)
        self.dark_theme = False
        if pn.state.location is not None:
            query_params = pn.state.location.query_params
            if 'theme' in query_params:
                self.dark_theme = (query_params['theme'] == 'dark')
            if self.dark_theme:
                pn.config.theme = 'dark'
            else:
                pn.config.theme = 'default'

        # Initialize controller and scene_config as None
        self.controller = None
        self.scene_config = None

        # Initialize interfaces and config_columns (will be populated when connected)
        self.interfaces = {}
        self.config_columns = pn.Row()  # Empty row initially

        # Create scene selection UI components
        self._setup_scene_selection_ui()

        # Create main container that will hold either scene selection or simulation UI
        self.main_container = pn.Column(sizing_mode="stretch_both")

        # Determine initial state and initialize appropriately
        if controller is not None:
            # Controller provided - initialize normally
            self._initialize_connected_ui(controller)
        elif check_server_running():
            # Server running - show scene selection with connect/stop options
            self._show_scene_selection()
        else:
            # No server - show scene selection
            self._show_scene_selection()

        # The app is always the main container
        self.app = self.main_container

    def _setup_scene_selection_ui(self):
        """Setup UI components for scene selection (shown when no server is running)."""
        # Get available scenes grouped by category
        scenes_grouped = get_available_scenes()

        # Build options dict with group structure for Select widget
        # Panel Select widget supports grouped options via nested dict
        scene_options = {}
        for category, scene_list in scenes_grouped.items():
            if scene_list:  # Only add non-empty categories
                for scene in scene_list:
                    scene_options[f"{category}: {scene}"] = scene

        # Scene selection widgets
        self.scene_select = pn.widgets.Select(
            name='Select Scene',
            options=scene_options,
            value=list(scene_options.values())[0] if scene_options else None,
            width=300,
        )

        self.start_server_btn = pn.widgets.Button(
            name='Start Simulation',
            button_type='success',
            width=200,
        )
        self.start_server_btn.on_click(self._start_server_cb)

        # Buttons for when a server is already running
        self.connect_existing_btn = pn.widgets.Button(
            name='Connect to Server',
            button_type='primary',
            width=200,
        )
        self.connect_existing_btn.on_click(self._connect_existing_cb)

        self.stop_existing_btn = pn.widgets.Button(
            name='Stop Server',
            button_type='warning',
            width=200,
        )
        self.stop_existing_btn.on_click(self._stop_existing_cb)

        self.server_status = pn.pane.Markdown(
            "### Select a scene to start the simulation",
            sizing_mode="stretch_width"
        )

        # Row for existing server options (hidden by default)
        self.existing_server_row = pn.Row(
            self.connect_existing_btn,
            self.stop_existing_btn,
            align="center",
            visible=False,
        )

        # Row for scene selection (hidden when server is running)
        self.scene_select_row = pn.Row(self.scene_select, align="center")
        self.start_server_row = pn.Row(self.start_server_btn, align="center")

        # Scene selection panel layout
        self.scene_selection_panel = pn.Column(
            pn.pane.Markdown("# Vivarium", align="center", styles={'font-size': '2em'}),
            pn.layout.Spacer(height=20),
            self.server_status,
            pn.layout.Spacer(height=20),
            self.existing_server_row,
            pn.layout.Spacer(height=10),
            self.scene_select_row,
            pn.layout.Spacer(height=10),
            self.start_server_row,
            pn.layout.Spacer(height=20),
            align="center",
            sizing_mode="stretch_both",
        )

    def _show_scene_selection(self, error_message=None):
        """Show the scene selection screen."""
        self.start_server_btn.disabled = False

        # Check if a server is already running
        if check_server_running():
            # Get the scene name from the running server
            try:
                client = SimulatorGRPCClient()
                running_scene = client.scene_name
                client.close()
                self.server_status.object = f"### Server is running with scene '{running_scene}'"
            except Exception as e:
                lg.warning(f"Could not get scene name from server: {e}")
                self.server_status.object = "### A server is running"
            # Show connect/stop options, hide scene selection
            self.existing_server_row.visible = True
            self.scene_select_row.visible = False
            self.start_server_row.visible = False
        elif error_message:
            self.server_status.object = f"### {error_message}"
            self.existing_server_row.visible = False
            self.scene_select_row.visible = True
            self.start_server_row.visible = True
        else:
            self.server_status.object = "### Select a scene to start the simulation"
            self.existing_server_row.visible = False
            self.scene_select_row.visible = True
            self.start_server_row.visible = True

        self.main_container.clear()
        self.main_container.append(self.scene_selection_panel)

    def _connect_existing_cb(self, event):
        """Callback to connect to an existing server."""
        try:
            client = SimulatorGRPCClient()
            controller = VivariumController(client=client)
            self._initialize_connected_ui(controller)
        except Exception as e:
            lg.error(f"Failed to connect to server: {e}")
            self._show_scene_selection(error_message=f"Failed to connect: {e}")

    def _stop_existing_cb(self, event):
        """Callback to stop the existing server."""
        _, server_pids = get_server_interface_pids()
        if server_pids:
            terminate_process(server_pids)
            lg.info("Server stopped")
        # Refresh the scene selection screen
        self._show_scene_selection()

    def _initialize_connected_ui(self, controller):
        """Initialize the full simulation UI after connecting to a server."""
        self.controller = controller
        client = self.controller.client
        self.scene_config = load_scene_config(client.scene_name)

        # Update dark theme from scene config if not set via URL
        if pn.state.location is None or 'theme' not in pn.state.location.query_params:
            self.dark_theme = self.scene_config.interface.dark_mode
            if self.dark_theme:
                pn.config.theme = 'dark'

        self.use_streaming = self.scene_config.interface.use_streaming
        self.controller_names = list(self.controller.controllers.keys())

        self.interfaces = create_interfaces(
            self.scene_config.environment.components.component_list,
            self.controller.controllers,
            self.controller.client.state,
            panel_cls=pn.Column
        )

        for name, interface in self.interfaces.items():
            interface.udpate_other_interfaces(self.interfaces)

        # TODO: (2025-08-26) move this to a dedicated SimulatorInterface class?
        self.param_simulator = ParamSimulator(self.controller.controllers['simulator'])
        self.param_simulator.update_from_server = True

        # Create simulation control widgets
        self._setup_simulation_widgets()

        # Create notebook widgets
        self._setup_notebook_widgets()

        self.plot = self.create_plot()

        # Build and show the simulation UI
        simulation_ui = self._create_simulation_ui()
        self.main_container.clear()
        self.main_container.append(simulation_ui)

        if not self.testing_mode:
            self.set_callbacks()
            # Start streaming if enabled
            if self.use_streaming:
                self._start_streaming()
        self.update_plot_cb()

    def _setup_simulation_widgets(self):
        """Setup widgets for simulation control."""
        self.start_toggle = pn.widgets.Toggle(
            **(
                {"name": "Pause simulator", "value": True}
                if self.controller.simulator.simulation_running
                else {"name": "Start simulator", "value": False}
            ),
            align="center",
        )

        self.plot_fps = pn.widgets.FloatInput(
            name="Plot FPS", value=15, width=80
        )

        # Currently not displayed
        self.streaming_toggle = pn.widgets.Toggle(
            name="Use Streaming" if not self.use_streaming else "Using Streaming",
            value=self.use_streaming,
            align="center",
        )

        self.drag_n_drop = pn.widgets.Toggle(name="Start Drag & Drop", value=False, align="center")

        self.dark_theme_switch = pn.widgets.Switch(
            name="Light/Dark theme",
            value=self.dark_theme,
            align="center",
        )

        self.controller_toggle = pn.widgets.ToggleGroup(
            name="ControllerToggle",
            options=self.controller_names,
            align="center",
            value=self.controller_names,
        )

        # Stop server button
        self.stop_server_btn = pn.widgets.Button(
            name='Stop Server',
            button_type='warning',
            width=120,
        )
        self.stop_server_btn.on_click(self._stop_server_cb)

        # Confirmation dialog widgets
        self.stop_confirm_panel = pn.Column(
            pn.pane.Markdown("### Are you sure you want to stop the server?"),
            pn.pane.Markdown("This will disconnect all clients."),
            pn.Row(
                pn.widgets.Button(name="Yes, Stop", button_type="danger", width=100),
                pn.widgets.Button(name="Cancel", button_type="default", width=100),
            ),
            visible=False,
        )
        # Wire up confirmation buttons
        self.stop_confirm_panel[2][0].on_click(self._confirm_stop_cb)
        self.stop_confirm_panel[2][1].on_click(self._cancel_stop_cb)

    def _setup_notebook_widgets(self):
        """Setup widgets for notebook/Jupyter control."""
        # Notebook configuration - load from config
        notebook_config = getattr(self.scene_config.interface, 'notebook', None)
        self.notebook_path = None
        self.jupyter_port = 8889

        if notebook_config is not None:
            if hasattr(notebook_config, 'path'):
                self.notebook_path = notebook_config.path
            if hasattr(notebook_config, 'jupyter_port'):
                self.jupyter_port = notebook_config.jupyter_port

        # Jupyter server management
        self.jupyter_process = None
        # Track whether we started this Jupyter server (vs connecting to external)
        self._jupyter_started_by_us = False

        # UI for Jupyter control and notebook display
        self.jupyter_status = pn.pane.Markdown("**Jupyter Status:** Checking...", sizing_mode="stretch_width")

        self.jupyter_port_input = pn.widgets.IntInput(
            name="Jupyter Port:",
            value=self.jupyter_port,
            start=8888,
            end=9999,
            step=1,
            width=120,
        )

        # Check Server button for manual status check
        self.check_jupyter_btn = pn.widgets.Button(
            name="Check Server",
            button_type="default",
            width=120,
        )

        self.start_jupyter_btn = pn.widgets.Button(
            name="Start Jupyter Server",
            button_type="success",
            width=200,
        )

        self.stop_jupyter_btn = pn.widgets.Button(
            name="Stop Jupyter Server",
            button_type="danger",
            width=200,
            visible=False,
        )

        self.open_configured_notebook_btn = pn.widgets.Button(
            name=f"Open {os.path.basename(self.notebook_path) if self.notebook_path else 'Configured Notebook'}",
            button_type="primary",
            width=250,
            visible=False,
        )

        self.open_new_notebook_btn = pn.widgets.Button(
            name="Open New Notebook",
            button_type="primary",
            width=200,
            visible=False,
        )

        self.notebook_url = pn.widgets.TextInput(
            name="Or enter notebook URL:",
            placeholder=f"http://localhost:{self.jupyter_port}/notebooks/path/to/notebook.ipynb",
            value="",
            width=400,
            visible=False,
        )

        # Conflict resolution panel (hidden by default)
        self._conflict_port = None  # Track which port has the conflict
        self._conflict_message = pn.pane.Markdown("")
        self._suggested_port_msg = pn.pane.Markdown("")

        self._jupyter_use_existing_btn = pn.widgets.Button(
            name="Use Existing", button_type="primary", width=120
        )
        self._jupyter_kill_only_btn = pn.widgets.Button(
            name="Kill Server", button_type="danger", width=100
        )
        self._jupyter_kill_restart_btn = pn.widgets.Button(
            name="Kill & Restart", button_type="warning", width=120
        )
        self._jupyter_use_different_port_btn = pn.widgets.Button(
            name="Use Different Port", button_type="success", width=150
        )
        self._jupyter_cancel_conflict_btn = pn.widgets.Button(
            name="Cancel", button_type="default", width=80
        )

        self.jupyter_conflict_panel = pn.Column(
            pn.pane.Markdown("### Jupyter Server Already Running", styles={'color': 'orange'}),
            self._conflict_message,
            pn.Row(
                self._jupyter_use_existing_btn,
                self._jupyter_kill_only_btn,
                self._jupyter_kill_restart_btn,
                self._jupyter_use_different_port_btn,
            ),
            pn.Row(
                self._jupyter_cancel_conflict_btn,
            ),
            self._suggested_port_msg,
            visible=False,
        )

    def _start_server_cb(self, event):
        """Callback for starting the simulation server."""
        scene_name = self.scene_select.value
        if scene_name is None:
            self.server_status.object = "### Please select a scene"
            return

        self.server_status.object = f"### Starting simulation with scene '{scene_name}'..."
        self.start_server_btn.disabled = True

        try:
            # Start the server and get a controller (controller manages server lifecycle)
            controller = VivariumController(start_server=True, scene_name=scene_name, timeout=self.server_timeout)
            self._started_server = True

            # Transition to full simulation UI
            self._initialize_connected_ui(controller)

        except Exception as e:
            lg.error(f"Failed to start server: {e}")
            self.server_status.object = f"### Error: {e}"
            self.start_server_btn.disabled = False
            self._started_server = False

    def _stop_server_cb(self, event):
        """Callback for the stop server button - shows confirmation."""
        self.stop_confirm_panel.visible = True

    def _cancel_stop_cb(self, event):
        """Callback to cancel stopping the server."""
        self.stop_confirm_panel.visible = False

    def _confirm_stop_cb(self, event):
        """Callback to confirm stopping the server."""
        self.stop_confirm_panel.visible = False

        # Stop the periodic callback
        if hasattr(self, 'pcb_plot') and self.pcb_plot.running:
            self.pcb_plot.stop()

        # Stop streaming if active
        if self._streaming_active:
            self._stop_streaming()

        # Disconnect the controller (don't close - we'll stop server separately)
        if self.controller:
            try:
                self.controller.disconnect()
            except Exception as e:
                lg.warning(f"Error disconnecting controller: {e}")
            self.controller = None

        # Stop the server (regardless of who started it)
        server_pids = kill_vivarium_processes(server=True)
        if server_pids:
            lg.info("Server stopped")

        # Reset state
        self.scene_config = None
        self.interfaces = {}
        self._started_server = False

        # Return to scene selection
        self._show_scene_selection()

    def start_toggle_cb(self, event):
        """Callback for the start/stop button

        :param event: The event for the new value of the button
        """
        self.controller.simulator.simulation_running = event.new
        self.start_toggle.name = "Pause simulator" if event.new else "Start simulator"

    def controller_toggle_cb(self, event):
        for cc in self.config_columns:
            cc.visible = cc.name in event.new

    def update_plot_fps(self, event):
        if event.new > 0:
            self.pcb_plot.period = int((1.0 / event.new) * 1000)
            if not self.pcb_plot.running:
                self.pcb_plot.start()
            # Also update streaming FPS if active
            if self._streaming_active:
                self._restart_streaming_with_fps(event.new)
        else:
            self.pcb_plot.stop()

    def _on_state_stream_update(self, state_and_cp):
        """Callback for state updates from streaming.
        
        This runs in a background thread, so we just set a flag
        and let the periodic callback handle the actual UI update.
        """
        with self._state_lock:
            # State is already updated in client by the streaming callback
            self._pending_state_update.set()

    def _start_streaming(self):
        """Start receiving state updates via streaming."""
        if self._streaming_active:
            return
        
        # Only start streaming if we have a gRPC client
        client = self.controller.client
        if hasattr(client, 'start_state_stream'):
            max_fps = int(self.plot_fps.value) if self.plot_fps.value > 0 else 30
            client.start_state_stream(
                callback=self._on_state_stream_update,
                max_fps=max_fps,
                include_controller_params=True
            )
            self._streaming_active = True
            lg.info(f"Started state streaming at max {max_fps} FPS")
        else:
            lg.warning("Client does not support streaming, falling back to polling")

    def _stop_streaming(self):
        """Stop receiving state updates via streaming."""
        if not self._streaming_active:
            return

        if self.controller is None:
            self._streaming_active = False
            return

        client = self.controller.client
        if hasattr(client, 'stop_state_stream'):
            client.stop_state_stream()
            self._streaming_active = False
            lg.info("Stopped state streaming")

    def _restart_streaming_with_fps(self, new_fps):
        """Restart streaming with a new FPS limit."""
        self._stop_streaming()
        self._start_streaming()

    def streaming_toggle_cb(self, event):
        """Callback for the streaming toggle."""
        if event.new:
            self._start_streaming()
            self.streaming_toggle.name = "Using Streaming"
        else:
            self._stop_streaming()
            self.streaming_toggle.name = "Use Streaming"

    def update_plot_cb(self):
        """Periodic callback for the plot update"""
        # Guard against being called when not connected
        if self.controller is None:
            return

        for interface in self.interfaces.values():
            if interface.renderer is not None:
                interface.renderer.update()
        if self.apply_changes:
            self.controller.apply_changes()
        state = self.controller.client.state
        # if self.param_simulator.config_update:  # TODO: (2025-08-26) To change
        #     self.controller.pull_selected_entities()
        for interface in self.interfaces.values():
            renderer = interface.renderer
            if renderer is not None:
                renderer.update_cds(state)
            
    def drag_n_drop_cb(self, event):
        if event.new:
            self.plot.toolbar.active_tap = self.point_draw_tool
            self.start_toggle.value = False
            self.controller.apply_changes()
            if self.pcb_plot.running:
                self.pcb_plot.stop()
            self.drag_n_drop.name = "Stop Drag & Drop"
        else:
            self.plot.toolbar.active_tap = None
            if not self.pcb_plot.running:
                self.pcb_plot.start()
            self.start_toggle.value = True
            self.drag_n_drop.name = "Start Drag & Drop"

    def dark_theme_switch_cb(self, event):
        """Callback for the dark mode toggle button

        :param event: The event for the new value of the button (True if dark theme)
        """
        self.pcb_plot.stop()
        self.controller.close()
        del self.controller
        self.dark_theme = event.new
        if event.new:
            pn.state.location.param.update(search="?theme=dark")
        else:
            pn.state.location.param.update(search="?theme=light")

        pn.state.location.reload=False
        pn.state.location.reload=True

    def start_jupyter_cb(self, event):
        """Callback for starting Jupyter server"""
        port = self.jupyter_port_input.value

        # Check if port is already in use - show conflict panel if so
        if check_jupyter_running(port):
            self._show_jupyter_conflict_panel(port)
            return

        self._do_start_jupyter(port)

    def _do_start_jupyter(self, port):
        """Actually start the Jupyter server on the given port."""

        project_root = get_bundle_root()

        try:
            lg.info(f"Starting Jupyter server on port {port}...")
            self.jupyter_status.object = f"**Jupyter Status:** 🔄 Starting on port {port}..."

            self.jupyter_process = start_jupyter_server(
                port=port,
                notebook_dir=project_root,
                show_output=False,
                return_process_object=True
            )

            # Update the configured port to match what was actually used
            self.jupyter_port = port
            self.jupyter_port_input.value = port

            # Register this port as started by us (for cleanup tracking)
            register_jupyter_port(port)
            self._jupyter_started_by_us = True

            # Wait for Jupyter to fully start and bind to the port
            # PyInstaller builds may take longer to initialize
            for _ in range(20):
                time.sleep(0.5)
                if check_jupyter_running(port):
                    lg.info(f"Jupyter server confirmed running on port {port}")
                    break

            # Update UI to reflect running state
            self._check_jupyter_status()
        except RuntimeError as e:
            # Port became busy between check and start - show conflict panel
            if "already in use" in str(e).lower():
                self._show_jupyter_conflict_panel(port)
            else:
                error_msg = str(e)
                lg.error(error_msg)
                self.jupyter_status.object = f"**Jupyter Status:** ❌ {error_msg}"

    def _show_jupyter_conflict_panel(self, port):
        """Display the conflict resolution panel for the given port."""
        self._conflict_port = port
        self._conflict_message.object = f"A Jupyter server is already running on port **{port}**. What would you like to do?"

        # Find next available port for suggestion
        try:
            suggested = find_next_available_port(port + 1)
            self._suggested_port_msg.object = f"*Suggested available port: **{suggested}***"
        except RuntimeError:
            self._suggested_port_msg.object = "*No available ports found nearby*"

        self.jupyter_conflict_panel.visible = True

    def _jupyter_use_existing_cb(self, event):
        """Connect to the existing external Jupyter server without tracking it."""
        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        # Update port and status - we're using an external server
        self.jupyter_port = port
        self.jupyter_port_input.value = port
        self._jupyter_started_by_us = False  # Don't track - it's external
        self.jupyter_process = None

        lg.info(f"Using existing Jupyter server on port {port}")
        self._check_jupyter_status()

    def _jupyter_kill_only_cb(self, event):
        """Kill the existing Jupyter server without starting a new one."""
        from vivarium.utils.handle_server_interface import kill_port_processes

        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        lg.info(f"Killing Jupyter server on port {port}...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Stopping server on port {port}..."

        killed = kill_port_processes(port, servers_only=True)
        if killed:
            lg.info(f"Killed Jupyter processes: {killed}")

        # Unregister if it was tracked
        unregister_jupyter_port(port)

        # Small delay to let port free up
        time.sleep(0.5)

        # Update UI to reflect stopped state
        self._check_jupyter_status()

    def _jupyter_kill_restart_cb(self, event):
        """Kill existing server and start a new one (tracked)."""
        from vivarium.utils.handle_server_interface import kill_port_processes

        port = self._conflict_port
        self.jupyter_conflict_panel.visible = False

        lg.info(f"Killing existing Jupyter on port {port} and restarting...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Restarting on port {port}..."

        # Kill existing
        killed = kill_port_processes(port, servers_only=True)
        if killed:
            lg.info(f"Killed existing Jupyter processes: {killed}")

        # Unregister old if tracked
        unregister_jupyter_port(port)

        # Small delay to let port free up
        time.sleep(0.5)

        # Start new
        self._do_start_jupyter(port)

    def _jupyter_use_different_port_cb(self, event):
        """Use the next available port."""
        self.jupyter_conflict_panel.visible = False

        try:
            port = self._conflict_port
            suggested = find_next_available_port(port + 1)
            lg.info(f"Using alternative port {suggested}")
            self._do_start_jupyter(suggested)
        except RuntimeError as e:
            lg.error(f"Could not find available port: {e}")
            self.jupyter_status.object = f"**Jupyter Status:** ❌ {e}"

    def _jupyter_cancel_conflict_cb(self, event):
        """Cancel and do nothing."""
        self.jupyter_conflict_panel.visible = False
        self._check_jupyter_status()

    def check_jupyter_cb(self, event):
        """Manual check button handler - update Jupyter status display."""
        port = self.jupyter_port_input.value
        self.jupyter_port = port
        self._check_jupyter_status()

    def stop_jupyter_cb(self, event):
        """Callback for stopping Jupyter server"""
        from vivarium.utils.handle_server_interface import stop_jupyter_server

        lg.info("Stopping Jupyter server...")
        self.jupyter_status.object = f"**Jupyter Status:** 🔄 Stopping..."

        stop_jupyter_server(self.jupyter_process, port=self.jupyter_port)

        # Unregister from tracking and reset state
        unregister_jupyter_port(self.jupyter_port)
        self.jupyter_process = None
        self._jupyter_started_by_us = False

        # Update UI to reflect stopped state
        self._check_jupyter_status()

    def open_configured_notebook_cb(self, event):
        """Callback for opening the configured notebook"""
        if self.notebook_path:
            project_root = get_bundle_root()
            if not os.path.isabs(self.notebook_path):
                notebook_path = os.path.join(project_root, self.notebook_path)
            else:
                notebook_path = self.notebook_path

            notebook_rel_path = os.path.relpath(notebook_path, project_root)
            url = f"http://localhost:{self.jupyter_port}/notebooks/{notebook_rel_path}"
            self.notebook_url.value = url
            self._update_notebook()

    def open_new_notebook_cb(self, event):
        """Callback for opening a new notebook"""
        # Open the Jupyter tree/home page in the iframe
        url = f"http://localhost:{self.jupyter_port}/tree"
        self.notebook_url.value = url
        self._update_notebook()

    def notebook_url_cb(self, event):
        """Callback for when the notebook URL changes

        :param event: The event for the new URL value
        """
        if event.new:
            self._update_notebook()

    def _check_jupyter_status(self):
        """Check if Jupyter server is running and update UI accordingly"""
        from vivarium.utils.handle_server_interface import check_jupyter_running

        is_running = check_jupyter_running(self.jupyter_port)

        if is_running:
            self.jupyter_status.object = f"**Jupyter Status:** ✓ Running on port {self.jupyter_port}"
            self.jupyter_port_input.visible = False
            self.start_jupyter_btn.visible = False
            self.stop_jupyter_btn.visible = True
            self.open_configured_notebook_btn.visible = bool(self.notebook_path)
            self.open_new_notebook_btn.visible = True
            self.notebook_url.visible = True
        else:
            self.jupyter_status.object = f"**Jupyter Status:** ✗ Not running"
            self.jupyter_port_input.visible = True
            self.start_jupyter_btn.visible = True
            self.stop_jupyter_btn.visible = False
            self.open_configured_notebook_btn.visible = False
            self.open_new_notebook_btn.visible = False
            self.notebook_url.visible = False
            self.notebook_iframe.object = ""

    def _update_notebook(self):
        """Update the notebook iframe with the current URL"""
        url = self.notebook_url.value
        if url:
            # Allow scripts, forms, and same-origin for full Jupyter functionality
            iframe_html = f'''<iframe
                src="{url}"
                width="100%"
                height="100%"
                frameborder="0"
                style="border: 1px solid #ddd;"
                sandbox="allow-same-origin allow-scripts allow-forms allow-modals allow-popups allow-downloads"
                allow="clipboard-read; clipboard-write"
            ></iframe>'''
            self.notebook_iframe.object = iframe_html
        else:
            self.notebook_iframe.object = ""

    def create_plot(self):
        """Creates a bokeh plot for the simulator

        :return: A bokeh plot
        """
        if self.dark_theme:
            self.curdoc.theme = 'dark_minimal'

        p_tools = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"
        p = figure(tools=p_tools, active_drag="box_select")
        # p.axis.major_label_text_font_size = "24px"
        p.axis.visible = True
        p.grid.visible = False
        hover = HoverTool(tooltips=None)
        p.add_tools(hover)
        p.x_range = Range1d(0, self.controller.controllers['simulator'].env.box_size)
        p.y_range = Range1d(0, self.controller.controllers['simulator'].env.box_size)
        self.point_draw_tool = PointDrawTool(
            renderers=[interface.renderer.plot(p) for interface in self.interfaces.values() if interface.renderer is not None and interface.renderer.use_point_draw_tool],
            add=False,
        )
        p.add_tools(self.point_draw_tool)
        for interface in self.interfaces.values():
            if not (interface.renderer is None or interface.renderer.use_point_draw_tool):
                interface.renderer.plot(p)
        return p

    def _create_simulation_ui(self):
        """Creates the simulation UI panel.

        :return: the simulation UI panel
        """
        self.config_columns = pn.Row(
            *[
                pn.Column(
                    pn.pane.Markdown("### Simulator", align="center"),
                    pn.panel(self.param_simulator, name="Configuration",
                             widgets={param_name: {'width': 100, 'min_width': 80, 'max_width': 140} for param_name in self.param_simulator.param_names()}),
                    visible=True,
                    sizing_mode="stretch_height",
                    scroll=True,
                    name="SIMULATOR",
                )
            ]
            + [interface.widget for interface in self.interfaces.values()]
        )

        # Create the notebook iframe (initially empty)
        self.notebook_iframe = pn.pane.HTML(
            "",
            sizing_mode="stretch_both",
        )

        # Build tabs for the right side
        tabs_list = [
            ("Configurations", pn.Column(
                pn.Row("### Show Configurations", self.controller_toggle),
                pn.Row(*self.config_columns),
                sizing_mode="stretch_both",
            )),
            ("Notebook", pn.Column(
                self.jupyter_status,
                pn.Row(
                    self.jupyter_port_input,
                    self.check_jupyter_btn,
                    self.start_jupyter_btn,
                    self.stop_jupyter_btn,
                ),
                self.jupyter_conflict_panel,
                pn.Row(
                    self.open_configured_notebook_btn,
                    self.open_new_notebook_btn,
                ),
                self.notebook_url,
                self.notebook_iframe,
                sizing_mode="stretch_both",
            ))
        ]

        right_side_tabs = pn.Tabs(*tabs_list, sizing_mode="stretch_both")

        # Build the simulation UI
        simulation_ui = pn.Row(
            pn.Column(
                pn.Row(
                    self.start_toggle,
                    self.plot_fps,
                    # self.streaming_toggle,
                    self.drag_n_drop,
                    self.dark_theme_switch,
                    pn.layout.Spacer(width=20),
                    self.stop_server_btn,
                    self.stop_confirm_panel,
                ),
                pn.panel(self.plot, sizing_mode="scale_width"),
            ),
            right_side_tabs,
        )
        return simulation_ui

    def set_callbacks(self):
        """
        Set the callbacks for all the widgets in the app
        """
        # putting directly the slider value causes bugs on some OS
        self.pcb_plot = pn.state.add_periodic_callback(
            self.update_plot_cb, int((1. / self.plot_fps.value) * 1000)
        )
        self.controller_toggle.param.watch(self.controller_toggle_cb, "value")
        self.start_toggle.param.watch(self.start_toggle_cb, "value")
        self.plot_fps.param.watch(self.update_plot_fps, "value")
        # self.streaming_toggle.param.watch(self.streaming_toggle_cb, "value")
        self.drag_n_drop.param.watch(self.drag_n_drop_cb, "value")
        self.dark_theme_switch.param.watch(self.dark_theme_switch_cb, "value")
        # Notebook callbacks
        self.check_jupyter_btn.on_click(self.check_jupyter_cb)
        self.start_jupyter_btn.on_click(self.start_jupyter_cb)
        self.stop_jupyter_btn.on_click(self.stop_jupyter_cb)
        self.open_configured_notebook_btn.on_click(self.open_configured_notebook_cb)
        self.open_new_notebook_btn.on_click(self.open_new_notebook_cb)
        self.notebook_url.param.watch(self.notebook_url_cb, "value")
        # Conflict resolution callbacks
        self._jupyter_use_existing_btn.on_click(self._jupyter_use_existing_cb)
        self._jupyter_kill_only_btn.on_click(self._jupyter_kill_only_cb)
        self._jupyter_kill_restart_btn.on_click(self._jupyter_kill_restart_cb)
        self._jupyter_use_different_port_btn.on_click(self._jupyter_use_different_port_cb)
        self._jupyter_cancel_conflict_btn.on_click(self._jupyter_cancel_conflict_cb)


if __name__ == "__main__":
    wm = WindowManager()
    wm.app.servable()
