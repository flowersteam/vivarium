import hydra
import logging
import panel as pn

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    PointDrawTool,
    HoverTool,
    Range1d,
)
from param import Parameterized

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.utils.scene_configs import load_scene_config
from vivarium.controllers import SimulatorController


lg = logging.getLogger(__name__)


class WindowManager(Parameterized):

    update_switch = pn.widgets.Switch(name="Update plot", value=True, align="center")
    update_timestep = pn.widgets.IntSlider(
        name="Timestep (ms)", value=40, start=1, end=1000
    )

    def __init__(self, client=None, notebook_mode=False, testing_mode=False, **kwargs):
        super().__init__(**kwargs)
        pn.config.theme = 'dark'
        client = client or SimulatorGRPCClient()
        self.scene_config = load_scene_config(client.scene_name)
        self.controller = SimulatorController.from_config(self.scene_config.environment.components, client=client)
        self.entity_types = list(self.controller.controllers.keys())
        
        self.interfaces = {}
        for name, component in self.scene_config.environment.components.component_list.items():
            if 'client' in component:
                if 'interface_cls' in component.client:
                    interface_cls = hydra.utils.get_class(component.client.interface_cls)
                    self.interfaces[name] = interface_cls(
                        self.controller.controllers[name],
                        self.controller.state,
                        self.controller.subtype_labels,
                        panel_cls=pn.Column
                    )

        self.start_toggle = pn.widgets.Toggle(
            **(
                {"name": "Stop", "value": True}
                if self.controller.is_started()
                else {"name": "Start", "value": False}
            ),
            align="center",
        )
        self.entity_toggle = pn.widgets.ToggleGroup(
            name="EntityToggle",
            options=self.entity_types,
            align="center",
            value=self.entity_types,
        )
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
        if event.new != self.controller.is_started():
            if event.new:
                self.controller.start()
            else:
                self.controller.stop()
        self.start_toggle.name = "Stop" if self.controller.is_started() else "Start"

    def entity_toggle_cb(self, event):
        for cc in self.config_columns:
            cc.visible = cc.name in event.new

    def update_timestep_cb(self, event):
        """Callback for the timestep of the plot update

        :param event: The event for the new value of the timestep
        """
        self.pcb_plot.period = event.new

    def update_plot_cb(self):
        """Periodic callback for the plot update"""
        for interface in self.interfaces.values():
            interface.renderer.update()
        self.controller.apply_changes()
        state = self.controller.update_state()
        if self.controller.param_simulator.config_update:  # TODO: (2025-08-26) To change
            self.controller.pull_selected_entities()
        for interface in self.interfaces.values():
            renderer = interface.renderer
            with renderer.no_drag_cb():
                renderer.update_cds(state)

    def update_switch_cb(self, event):
        """Callback for the plot update switch

        :param event: The event for the new value of the switch
        """
        if event.new and not self.pcb_plot.running:
            self.pcb_plot.start()
        elif not event.new and self.pcb_plot.running:
            self.pcb_plot.stop()

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
        p.x_range = Range1d(0, self.controller.param_simulator.box_size)
        p.y_range = Range1d(0, self.controller.param_simulator.box_size)
        draw_tool = PointDrawTool(
            renderers=[self.interfaces[etype].renderer.plot(p) for etype in self.entity_types],
            add=False,
        )
        p.add_tools(draw_tool)
        return p

    def create_app(self):
        """Creates a panel app

        :return: the panel app
        """
        self.config_columns = pn.Row(
            *[
                pn.Column(
                    pn.pane.Markdown("### SIMULATOR", align="center"),
                    pn.panel(self.controller.param_simulator, name="Configuration"),
                    visible=True,
                    sizing_mode="scale_height",
                    scroll=True,
                    name="SIMULATOR",
                )
            ]
            + [self.interfaces[etype].widget for etype in self.entity_types]
        )

        app = pn.Row(
            pn.Column(
                (
                    pn.Row(
                        pn.pane.Markdown("### Start/Stop server", align="center"),
                        self.start_toggle,
                    )
                    if not self.notebook_mode
                    else None
                ),
                pn.Row(
                    pn.pane.Markdown("### Start/Stop update", align="center"),
                    self.update_switch,
                    self.update_timestep,
                ),
                pn.panel(self.plot),
            ),
            pn.Column(
                pn.Row("### Show Configurations", self.entity_toggle),
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
            self.update_plot_cb, self.update_timestep.value
        )
        self.entity_toggle.param.watch(self.entity_toggle_cb, "value")
        self.start_toggle.param.watch(self.start_toggle_cb, "value")
        self.update_switch.param.watch(self.update_switch_cb, "value")
        self.update_timestep.param.watch(self.update_timestep_cb, "value")


if __name__ == "__main__":
    wm = WindowManager()
    wm.app.servable()
