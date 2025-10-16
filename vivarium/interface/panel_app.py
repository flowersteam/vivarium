import hydra
import logging
import panel as pn
from param import Parameterized

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    PointDrawTool,
    HoverTool,
    Range1d,
)

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import ParamSimulator
from vivarium.utils.scene_configs import load_scene_config
from vivarium.controllers import SimulatorController


lg = logging.getLogger(__name__)

def create_interfaces(component_list_config, controllers, state, panel_cls=pn.Column):
    interfaces = {}
    for name, component in component_list_config.items():
        if 'client' in component:
            if 'interface_cls' in component.client:
                interface_cls = hydra.utils.get_class(component.client.interface_cls)
                interfaces[name] = interface_cls(
                    controllers[name],
                    state,
                    panel_cls=panel_cls
                )
    return interfaces


class WindowManager(Parameterized):


    def __init__(self, client=None, notebook_mode=False, testing_mode=False, **kwargs):
        super().__init__(**kwargs)
        pn.config.theme = 'dark'

        client = client or SimulatorGRPCClient()
        self.scene_config = load_scene_config(client.scene_name)
        self.controller = SimulatorController.from_config(self.scene_config.environment.components,
                                                          client=client,
                                                          run_from_server=self.scene_config.simulator.run_from_server)
        self.controller_names = list(self.controller.controllers.keys())
        
        self.interfaces = create_interfaces(
            self.scene_config.environment.components.component_list,
            self.controller.controllers,
            self.controller.state,
            panel_cls=pn.Column
        )
        
        # TODO: (2025-08-26) move this to a dedicated SimulatorInterface class?
        self.param_simulator = ParamSimulator(self.controller.simulator_parameters)
        self.param_simulator.update_from_server = True

        self.start_toggle = pn.widgets.Toggle(
            **(
                {"name": "Pause simulator", "value": True}
                if self.controller.is_running()
                else {"name": "Start simulator", "value": False}
            ),
            align="center",
        )
        self.plot_fps = pn.widgets.FloatInput(
            name="Plot FPS", value=25, width=80
        )
        self.drag_n_drop = pn.widgets.Toggle(name="Start Drag & Drop", value=False, align="center")
        self.controller_toggle = pn.widgets.ToggleGroup(
            name="ControllerToggle",
            options=self.controller_names,
            align="center",
            value=self.controller_names,
        )
        
        #TODO: Obsolete, to remove here and all other modules using it
        self.notebook_mode = notebook_mode

        self.plot = self.create_plot()
        self.app = self.create_app()
        if not testing_mode:
            self.set_callbacks()
        self.update_plot_cb()

    def start_toggle_cb(self, event):
        """Callback for the start/stop button

        :param event: The event for the new value of the button
        """
        if event.new != self.controller.is_running():
            if event.new:
                self.controller.run()
            else:
                self.controller.stop()
        self.start_toggle.name = "Pause simulator" if self.controller.is_running() else "Start simulator"

    def controller_toggle_cb(self, event):
        for cc in self.config_columns:
            cc.visible = cc.name in event.new

    def update_plot_fps(self, event):
        if event.new > 0:
            self.pcb_plot.period = int((1.0 / event.new) * 1000)
            if not self.pcb_plot.running:
                self.pcb_plot.start()
        else:
            self.pcb_plot.stop()

    def update_plot_cb(self):
        """Periodic callback for the plot update"""
        for interface in self.interfaces.values():
            if interface.renderer is not None:
                interface.renderer.update()
        self.controller.apply_changes()
        state = self.controller.update_state()
        # if self.param_simulator.config_update:  # TODO: (2025-08-26) To change
        #     self.controller.pull_selected_entities()
        for interface in self.interfaces.values():
            renderer = interface.renderer
            if renderer is not None:
                renderer.update_cds(state)
            
    def drag_n_drop_cb(self, event):
        if event.new:
            self.plot.toolbar.active_tap = self.point_draw_tool
            if self.controller.run_from_server:
                self.start_toggle.value = False
            if self.pcb_plot.running:
                self.pcb_plot.stop()
            self.drag_n_drop.name = "Stop Drag & Drop"
        else:
            self.plot.toolbar.active_tap = None
            if self.controller.run_from_server:
                self.start_toggle.value = True
            if not self.pcb_plot.running:
                self.pcb_plot.start()
            self.drag_n_drop.name = "Start Drag & Drop"

    def create_plot(self):
        """Creates a bokeh plot for the simulator

        :return: A bokeh plot
        """
        curdoc().theme = 'dark_minimal'

        p_tools = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"
        p = figure(tools=p_tools, active_drag="box_select")
        # p.axis.major_label_text_font_size = "24px"
        p.axis.visible = False
        p.grid.visible = False
        hover = HoverTool(tooltips=None)
        p.add_tools(hover)
        p.x_range = Range1d(0, self.controller.simulator_parameters.box_size)
        p.y_range = Range1d(0, self.controller.simulator_parameters.box_size)
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

        app = pn.Row(
            pn.Column(
                pn.Row(
                    self.start_toggle if self.controller.run_from_server else None,
                    self.plot_fps,
                    self.drag_n_drop,
                ),
                pn.panel(self.plot, sizing_mode="scale_width"),
            ),
            pn.Column(
                pn.Row("### Show Configurations", self.controller_toggle),
                pn.Row(*self.config_columns),
            ),
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
        # self.update_switch.param.watch(self.update_switch_cb, "value")
        self.plot_fps.param.watch(self.update_plot_fps, "value")
        self.drag_n_drop.param.watch(self.drag_n_drop_cb, "value")


if __name__ == "__main__":
    wm = WindowManager()
    wm.app.servable()
