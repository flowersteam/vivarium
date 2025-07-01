from enum import Enum
import logging
import panel as pn
from contextlib import contextmanager

import numpy as np

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    PointDrawTool,
    HoverTool,
    Range1d,
    CDSView,
    BooleanFilter,
)
from param import Parameterized

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import PanelController
from vivarium.utils.scene_configs import SceneConfiguration


lg = logging.getLogger(__name__)


class Shape(Enum):
    CIRCLE = 0
    SQUARE = 1


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


class EntityManager:
    def __init__(
        self, 
        entities,
        selected_param_entity,
        param_simulator_state,
        selected, etype, state,
        shape,
        line_width=1.0,
    ):
        self.entities = entities
        self.selected_param_entity = selected_param_entity
        self.param_simulator_state = param_simulator_state
        self.selected = selected
        self.etype = etype
        self.shape = getattr(Shape, shape.upper()) if isinstance(shape, str) else shape
        self.line_width = line_width
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        self.cds.on_change("data", self.drag_cb)
        self.cds_view = self.create_cds_view()
        self.param_simulator_state.param.watch(
            self.hide_all_non_existing, "hide_non_existing", onlychanged=True, 
        )
        selected.param.watch(
            self.update_selected_plot, ["selection"], onlychanged=True, precedence=0
        )
        self.selected_param_entity.param.watch(self.update_cds_view, 
                                               self.selected_param_entity.panel_visibility_parameters, 
                                               onlychanged=True)
        self.selected_param_entity.param.watch(self.hide_non_existing, 
                                               "exists", 
                                               onlychanged=True)
        self.apply_visible_filter()

    def drag_cb(self, attr, old, new):
        """Callback for the drag & drop of entities

        :param attr: (unused)
        :param old: (unused)
        :param new: The event containing the new positions of the entities
        """
        for i, e in enumerate(self.entities):
            e.x_position = new["x"][i]
            e.y_position = new["y"][i]

    @contextmanager
    def no_drag_cb(self):
        """Prevent the CDS from updating the configs when the change comes from the
        server
        """
        self.cds.remove_on_change("data", self.drag_cb)
        yield
        self.cds.on_change("data", self.drag_cb)

    def get_cds_data(self, state):
        """Update the ColumnDataSource with the new data

        :param state: The state coming from the server
        :return: Data dictionary for the ColumnDataSource
        """
        pos = state.position_center(self.etype)
        x, y = pos[:, 0], pos[:, 1]
        o = state.position_orientation(self.etype)
        d = state.diameter(self.etype)
        
        colors = [e.color for e in self.entities]

        data = dict(x=x, y=y, diameter=d, orientation=o, fill_color=colors)
        if self.shape == Shape.CIRCLE:
            data["radius"] = d / 2.0
        return data

    def update_cds(self, state):
        """Updates the ColumnDataSource with new data from server

        :param state: The state coming from the server
        """
        if self.param_simulator_state.hide_non_existing:
            exists = state.entity_state.exists[getattr(state, self.etype).entity_idx]
            for i, e in enumerate(self.entities):
                e.visible = bool(exists[i])
            self.apply_visible_filter()      
        self.cds.data.update(self.get_cds_data(state))

    def create_cds_view(self):
        """Creates a ColumnDataSource view for each visibility attribute

        :return: A dictionary of ColumnDataSource views for each visibility attribute
        """
        # For each panel attribute (i.e. visibility-related), create a filter
        # that is a logical AND of the visibility and the attribute
        return {
            attr: CDSView(
                filter=BooleanFilter(
                    [getattr(e, attr) and e.visible for e in self.entities]
                )
            )
            for attr in self.selected_param_entity.panel_visibility_parameters
        }

    def update_cds_view(self, event):
        """Updates the view of the ColumnDataSource if the visibility of an entity changes

        :param event: The event containing the changed value
        """
        n = event.name
        for attr in [n] if n != "visible" else self.selected_param_entity.panel_visibility_parameters:
            f = [getattr(e, attr) and e.visible for e in self.entities]
            self.cds_view[attr].filter = BooleanFilter(f)

    def update_selected_plot(self, event):
        """Updates the selected entities in the plot

        :param event: The event containing the new selected entities
        """
        self.cds.selected.indices = event.new

    def hide_all_non_existing(self, event):
        """Hides or shows all the entities that do not exist according to the global
        visibility of non-existing entities

        :param event: The event containing the new global "visibility of non-existing
        entities" value
        """
        for i, entity in enumerate(self.entities):
            if not entity.exists:
                entity.visible = not event.new
        
        # CMF added this
        self.apply_visible_filter()

    def apply_visible_filter(self):
        f = [e.visible for e in self.entities]
        for attr in self.selected_param_entity.panel_visibility_parameters:
            self.cds_view[attr].filter = BooleanFilter(f)

    def hide_non_existing(self, event):
        """Hides or shows an entity that does not exist depending on the global
        visibility of non-existing entities

        :param event: The event containing the new existence value
        """
        if not self.param_simulator_state.hide_non_existing:
            return
        self.selected_param_entity.visible = event.new


    def update_selected_simulator(self):
        """Updates the list of selected entities in the Selection list"""
        indices = self.cds.selected.indices
        if len(indices) > 0 and indices != self.selected.selection:
            self.selected.selection = indices

    def plot(self, fig: figure):
        """Plot the objects on the bokeh figure

        :param fig: A bokeh figure
        :return: The figure with the objects plotted
        """
        src = {"source": self.cds}
        
        if self.shape == Shape.CIRCLE:
            return fig.circle(
                "x",
                "y",
                radius="radius",
                fill_color="fill_color",
                fill_alpha=0.6,
                line_color="white",
                line_width=self.line_width,
                hover_fill_color="black",
                hover_fill_alpha=0.7,
                hover_line_color=None,
                view=self.cds_view["visible"],
                **src,
            )
        elif self.shape == Shape.SQUARE:
            return fig.rect(
                x="x",
                y="y",
                width="diameter",
                height="diameter",
                angle="orientation",
                fill_color="fill_color",
                fill_alpha=0.6,
                line_color="white",
                line_width=self.line_width,
                hover_fill_color="black",
                hover_fill_alpha=0.7,
                hover_line_color=None,
                view=self.cds_view["visible"],
                **src,
            )
        else:
            raise AttributeError('self.shape should be an instance of Shape')


class WindowManager(Parameterized):

    update_switch = pn.widgets.Switch(name="Update plot", value=True, align="center")
    update_timestep = pn.widgets.IntSlider(
        name="Timestep (ms)", value=40, start=1, end=1000
    )

    def __init__(self, client=None, notebook_mode=False, testing_mode=False, **kwargs):
        super().__init__(**kwargs)
        pn.config.theme = 'dark'
        client = client or SimulatorGRPCClient()
        self.scene_config = SceneConfiguration(client.scene_name)
        self.controller = PanelController(client=client, 
                                          subtypes=self.scene_config.config.subtypes, 
                                          **self.scene_config.create_controllers())
        self.entity_types = list(self.controller.entity_lists.keys())
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
        
        self.entity_manager_classes = {etype: c.render_cls for etype, c in self.controller.controllers.items()}
        self.entity_managers = {
            etype: manager_class(
                entities = self.controller.entity_lists[etype],
                selected_param_entity=self.controller.selected_entities[etype],
                param_simulator_state=self.controller.param_simulator,
                selected=self.controller.selected[etype],
                etype=etype,
                state=self.controller.state,
                ** self.scene_config.config.client[etype].renderer_kwargs
            )
            for etype, manager_class in self.entity_manager_classes.items()
        }

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
        for em in self.entity_managers.values():
            em.update_selected_simulator()
        self.controller.apply_changes()
        state = self.controller.update_state()
        if self.controller.param_simulator.config_update:
            self.controller.pull_selected_entities()
        for em in self.entity_managers.values():
            with em.no_drag_cb():
                em.update_cds(state)

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
            renderers=[self.entity_managers[etype].plot(p) for etype in self.entity_types],
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
            + [
                pn.Column(
                    pn.pane.Markdown(f"### {etype}", align="center"),
                    self.controller.selected[etype],
                    pn.panel(
                        self.controller.selected_entities[etype],
                        name="State configuration",
                    ),
                    visible=True,
                    sizing_mode="scale_height",
                    scroll=True,
                    name=etype,
                )
                for etype in self.entity_managers.keys()
            ]
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
