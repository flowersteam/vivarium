import logging
import threading

import hydra
import panel as pn
from param import Parameterized

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    PointDrawTool,
    HoverTool,
    Range1d,
)

from vivarium.controllers import VivariumController
from vivarium.utils.scene_configs import load_config, load_scene_config, get_available_scenes
from vivarium.runtime.paths import get_version, is_frozen, DEFAULT_JUPYTER_PORT
from vivarium.runtime import check_server_running, kill_vivarium_processes
from vivarium.runtime._process import get_server_interface_pids, terminate_process
from vivarium.interface.parameterized import ParamSimulator
from vivarium.interface.utils import cleanup_parameterized_class
from vivarium.interface.update_manager import UpdateManager
from vivarium.interface.jupyter_manager import JupyterManager
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient


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


class PanelApp(Parameterized):

    def __init__(self, controller=None, apply_changes=True, testing_mode=False, server_timeout=30.0, **kwargs):
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

        self.curdoc = curdoc()

        if pn.state.location is not None and 'theme' in pn.state.location.query_params:
            query_params = pn.state.location.query_params
            self.dark_theme = (query_params['theme'] == 'dark')
        else:
            self.dark_theme =  load_config("scene/interface", "base_interface").dark_mode

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

        # Create update manager (frozen/PyInstaller builds only)
        if is_frozen():
            self.update_manager = UpdateManager()
            self.update_manager.start_update_check()
        else:
            self.update_manager = None

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

        # Version display
        version = get_version()
        self.version_label = pn.pane.Markdown(
            f"v{version}",
            styles={'color': 'gray', 'font-size': '0.9em'},
            align="center",
        )

        self.dark_theme_switch = pn.widgets.Switch(
            name="Light/Dark theme",
            value=self.dark_theme,
            align="center",
        )

        self.dark_theme_switch.param.watch(self.dark_theme_switch_cb, "value")

        # Scene selection panel layout
        self.scene_selection_panel = pn.Column(
            pn.pane.Markdown("# Vivarium", align="center", styles={'font-size': '2em'}),
            self.version_label,
            pn.layout.Spacer(height=10),
            self.server_status,
            pn.layout.Spacer(height=20),
            self.existing_server_row,
            pn.layout.Spacer(height=10),
            self.scene_select_row,
            pn.layout.Spacer(height=10),
            self.start_server_row,
            pn.layout.Spacer(height=20),
            self.dark_theme_switch,
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

        # Insert update notifications if available (frozen builds only)
        if self.update_manager is not None:
            self.update_manager.insert_into(self.scene_selection_panel)

    def _connect_existing_cb(self, event):
        """Callback to connect to an existing server."""
        try:
            client = SimulatorGRPCClient()

            # controller.apply_changes() is already called in update_plot_cb and the state is fetched through the state streaming mechanism
            # So we don't need the controller thread in addition.
            controller = VivariumController(client=client, start_controller_thread=False)

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

        self.controller_names = list(self.controller.controllers.keys())

        self.interfaces = create_interfaces(
            self.scene_config.environment.components.component_list,
            self.controller.controllers,
            self.controller.client.state,
            panel_cls=pn.Column
        )

        for name, interface in self.interfaces.items():
            interface.update_other_interfaces(self.interfaces)

        # TODO: (2025-08-26) move this to a dedicated SimulatorInterface class?
        self.param_simulator = ParamSimulator(self.controller.controllers['simulator'])
        self.param_simulator.update_from_server = True

        # Create simulation control widgets
        self._setup_simulation_widgets()

        # Create Jupyter manager with notebook config
        notebook_config = getattr(self.scene_config.interface, 'notebook', None)
        notebook_path = None
        jupyter_port = DEFAULT_JUPYTER_PORT
        if notebook_config is not None:
            if hasattr(notebook_config, 'path'):
                notebook_path = notebook_config.path
            if hasattr(notebook_config, 'jupyter_port'):
                jupyter_port = notebook_config.jupyter_port

        self.jupyter_manager = JupyterManager(
            notebook_path=notebook_path,
            jupyter_port=jupyter_port,
        )

        self.plot = self.create_plot()

        # Build and show the simulation UI
        simulation_ui = self._create_simulation_ui()
        self.main_container.clear()
        self.main_container.append(simulation_ui)

        if not self.testing_mode:
            self.set_callbacks()
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
            name="Plot FPS", value=10, width=80
        )

        self.drag_n_drop = pn.widgets.Toggle(name="Start Drag & Drop", value=False, align="center")

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
            controller = VivariumController(start_server=True, scene_name=scene_name, timeout=self.server_timeout, start_controller_thread=False)
            self._started_server = True

            # Transition to full simulation UI
            self._initialize_connected_ui(controller)

        except Exception as e:
            lg.exception(f"Failed to start server: {e}")
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

        self.jupyter_manager.stop()

        # Close the controller (disconnects and stops server if we started it)
        if self.controller:
            try:
                self.controller.close()
            except Exception as e:
                lg.warning(f"Error closing controller: {e}")
            self.controller = None

        # Fallback: kill any remaining server processes by port
        server_pids = kill_vivarium_processes(server=True)
        if server_pids:
            lg.info(f"Fallback cleanup killed server process(es): {server_pids}")

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

    def update_plot_cb(self):
        """Periodic callback for the plot update.

        Threading note: state and controller_parameters are updated in separate
        assignments by the gRPC client (in set_changes or the streaming thread).
        This callback could read between the two, causing a brief visual
        inconsistency (e.g. new positions with stale config). CPython's GIL
        prevents torn reads of individual attributes.
        """
        # Guard against being called when not connected
        if self.controller is None:
            return

        # Always apply pending UI changes to the server
        if self.apply_changes:
            self.controller.apply_changes()

        # Skip repaint when streaming is active but no new state has arrived
        if self._streaming_active and not self._pending_state_update.is_set():
            return
        self._pending_state_update.clear()

        for interface in self.interfaces.values():
            if interface.renderer is not None:
                interface.renderer.update()
        state = self.controller.client.state
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

        self.dark_theme = event.new
        if event.new:
            pn.state.location.param.update(search="?theme=dark")
        else:
            pn.state.location.param.update(search="?theme=light")

        pn.state.location.reload=False
        pn.state.location.reload=True

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
                    pn.panel(self.param_simulator, name="Controller",
                             widgets={param_name: {'width': 100, 'min_width': 80, 'max_width': 140} for param_name in self.param_simulator.param_names()}),
                    visible=True,
                    sizing_mode="stretch_height",
                    scroll=True,
                    name="SIMULATOR",
                )
            ]
            + [interface.widget for interface in self.interfaces.values()]
        )

        # Build tabs for the right side
        notebook_config = getattr(self.scene_config.interface, 'notebook', None)

        tabs_list = [
            ("Controllers", pn.Column(
                pn.Row("### Show Controllers", self.controller_toggle),
                pn.Row(*self.config_columns),
                sizing_mode="stretch_both",
            )),
            ("Notebook", self.jupyter_manager.create_notebook_tab()),
        ]

        right_side_tabs = pn.Tabs(*tabs_list, sizing_mode="stretch_both")

        if hasattr(notebook_config, 'path') and notebook_config.path:
            right_side_tabs.active = 1
            self.start_toggle.visible = False
        else:
            right_side_tabs.active = 0

        # Build the simulation UI
        simulation_ui = pn.Row(
            pn.Column(
                pn.Row(
                    self.start_toggle,
                    self.plot_fps,
                    self.drag_n_drop,
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
        self.drag_n_drop.param.watch(self.drag_n_drop_cb, "value")
        # Jupyter callbacks
        self.jupyter_manager.set_callbacks()


if __name__ == "__main__":
    wm = PanelApp()
    wm.app.servable()
