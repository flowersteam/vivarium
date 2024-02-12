import numpy as np
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import PanelController
from vivarium.simulator.sim_computation import EntityType

import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, Range1d


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


class EntityManager:
    def __init__(self, config, selected, etype, state):
        self.config = config
        self.selected = selected
        self.etype = etype
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        selected.param.watch(self.update_selected_plot, ['selection'], onlychanged=True, precedence=0)

    def get_cds_data(self, state):
        raise NotImplementedError()

    def update_cds(self, state):
        self.cds.data.update(self.get_cds_data(state))

    def update_selected_plot(self, event):
        self.cds.selected.indices = event.new

    def update_selected_simulator(self):
        if len(self.cds.selected.indices) > 0 and self.cds.selected.indices != self.selected.selection:
            self.selected.selection = self.cds.selected.indices

    def plot(self, figure):
        raise NotImplementedError()


class AgentManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        radii = state.diameter(self.etype) / 2.
        colors = state.agent_state.color  # ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

        normals = normal(np.array(thetas))

        orientation_lines_x = [[xx, xx + r * n[0]] for xx, n, r in zip(x, normals, radii)]
        orientation_lines_y = [[yy, yy + r * n[1]] for yy, n, r in zip(y, normals, radii)]

        return dict(x=x, y=y, ox=orientation_lines_x, oy=orientation_lines_y, r=radii, fc=colors)

    def plot(self, figure):
        figure.multi_line('ox', 'oy', source=self.cds, color='black', line_width=1)
        return figure.circle('x', 'y', radius='r', fill_color='fc', fill_alpha=0.6, line_color=None,
                             hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=self.cds)

class ObjectManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        d = state.diameter(self.etype)
        colors = state.object_state.color

        return dict(x=x, y=y, width=d, height=d, angle=thetas, fill_color=colors)

    def plot(self, figure):
        return figure.rect(x='x', y='y', width='width', height='height', angle='angle', fill_color='fill_color', fill_alpha=0.6, line_color=None,
                           hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=self.cds)


pn.extension()

simulator = PanelController(client=SimulatorGRPCClient())
state = simulator.state

entity_types = [EntityType.AGENT, EntityType.OBJECT]

entity_manager_classes = {EntityType.AGENT: AgentManager, EntityType.OBJECT: ObjectManager}

entity_managers = {etype: manager_class(config=simulator.selected_configs[etype], selected=simulator.selected_entities[etype], etype=etype, state=state) for etype, manager_class in entity_manager_classes.items()}

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None)
p.add_tools(hover)
p.x_range = Range1d(0, simulator.simulator_config.box_size)
p.y_range = Range1d(0, simulator.simulator_config.box_size)

toggle = pn.widgets.Toggle(name="Stop" if simulator.is_started() else "Start")

def callback(event):
    if simulator.is_started():
        simulator.stop()
        toggle.name = "Start"
    else:
        simulator.start()
        toggle.name = "Stop"

toggle.param.watch(callback, "value")

draw_tool = PointDrawTool(renderers=[entity_managers[etype].plot(p) for etype in entity_types])
p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool


# https://panel.holoviz.org/how_to/param/custom.html
sim_panel = pn.Param(simulator.param)

# Selector for entity attributes
entity_columns = [pn.Column(simulator.selected_entities[etype], simulator.selected_configs[etype], visible=False) for etype in EntityType]

entity_toggle = pn.widgets.ToggleGroup(name="EntityToggle", options=[t.name for t in entity_types])

def toggle_callback(event):
    for i, t in enumerate(entity_types):
        entity_columns[i].visible = (t.name in event.new)

entity_toggle.param.watch(toggle_callback, "value")

# Switch for simulator settings
settings_pane = pn.panel(simulator.simulator_config, visible=False)

settings_switch = pn.widgets.Switch(name='Simulator settings')

def settings_callback(event):
    settings_pane.visible = event.new

settings_switch.param.watch(settings_callback, "value")

row = pn.Row(pn.Column(toggle, p, pn.Row("Show sim settings", settings_switch), settings_pane),
             pn.Column(pn.Row("### Show attributes",entity_toggle), pn.Row(*entity_columns)))

row.servable()


def update_plot():
    for em in entity_managers.values():
        em.update_selected_simulator()
    state = simulator.update_state()
    simulator.pull_configs()
    for em in entity_managers.values():
        em.update_cds(state)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)