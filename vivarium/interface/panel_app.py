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
from vivarium.utils.scene_configs import load_scene_config
from vivarium.controllers.panel_controller import ParamSimulator
from vivarium.interface.utils import cleanup_parameterized_class
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


class WindowManager(Parameterized):

    def __init__(self, controller=None, apply_changes=True, notebook_mode=False, testing_mode=False, jupyter_enabled=False, **kwargs):
        
        
        super().__init__(**kwargs)

        if controller is None:
            client = SimulatorGRPCClient()
            self.scene_config = load_scene_config(client.scene_name)
            self.controller = VivariumController.from_client(client=client)
        else:
            self.controller = controller
            client = self.controller.client
            self.scene_config = load_scene_config(client.scene_name)

        self.dark_theme = self.scene_config.interface.dark_mode
        if pn.state.location is not None:
            query_params = pn.state.location.query_params
            if 'theme' in query_params:
                self.dark_theme = (query_params['theme'] == 'dark')
            if self.dark_theme:
                pn.config.theme = 'dark'
            else:
                pn.config.theme = 'default'
        

        self.apply_changes = apply_changes
        self.use_streaming = self.scene_config.interface.use_streaming
        self._streaming_active = False
        self._pending_state_update = threading.Event()
        self._state_lock = threading.Lock()
        
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

        # Notebook iframe widgets - load from config
        # Only show notebook section if Jupyter is enabled
        self.jupyter_enabled = jupyter_enabled
        notebook_config = getattr(self.scene_config.interface, 'notebook', None)
        notebook_path = None
        jupyter_port = 8889

        if notebook_config is not None:
            if hasattr(notebook_config, 'path'):
                notebook_path = notebook_config.path
            if hasattr(notebook_config, 'jupyter_port'):
                jupyter_port = notebook_config.jupyter_port

        # Build notebook URL if path is specified and Jupyter is enabled
        default_url = ""
        if jupyter_enabled and notebook_path:
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            # Check if it's an absolute path or relative
            if not os.path.isabs(notebook_path):
                notebook_path = os.path.join(project_root, notebook_path)

            # Convert to URL path (relative to project root for Jupyter)
            notebook_rel_path = os.path.relpath(notebook_path, project_root)
            default_url = f"http://localhost:{jupyter_port}/notebooks/{notebook_rel_path}"

        self.notebook_url = pn.widgets.TextInput(
            name="Notebook URL",
            placeholder="http://localhost:8888/notebooks/path/to/notebook.ipynb",
            value=default_url,
            width=400,
        )
        
        #TODO: Obsolete, to remove here and all other modules using it
        self.notebook_mode = notebook_mode       

        self.curdoc = curdoc()
        
        self.plot = self.create_plot()
        self.app = self.create_app()
        if not testing_mode:
            self.set_callbacks()
            # Start streaming if enabled
            if self.use_streaming:
                self._start_streaming()
            # Load notebook if configured
            if self.jupyter_enabled and default_url:
                self._update_notebook()
        self.update_plot_cb()

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

    def notebook_url_cb(self, event):
        """Callback for when the notebook URL changes

        :param event: The event for the new URL value
        """
        if event.new:
            self._update_notebook()

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

    def create_app(self):
        """Creates a panel app

        :return: the panel app
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
            ))
        ]

        # Add notebook tab if Jupyter is enabled
        if self.jupyter_enabled:
            tabs_list.append(
                ("Notebook", pn.Column(
                    self.notebook_url,
                    self.notebook_iframe,
                    sizing_mode="stretch_both",
                ))
            )

        right_side_tabs = pn.Tabs(*tabs_list, sizing_mode="stretch_both")

        # Build the main app
        app = pn.Row(
            pn.Column(
                pn.Row(
                    self.start_toggle,
                    self.plot_fps,
                    # self.streaming_toggle,
                    self.drag_n_drop,
                    self.dark_theme_switch
                ),
                pn.panel(self.plot, sizing_mode="scale_width"),
            ),
            right_side_tabs,
        )
        return app

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
        self.notebook_url.param.watch(self.notebook_url_cb, "value")


if __name__ == "__main__":
    wm = WindowManager()
    wm.app.servable()
