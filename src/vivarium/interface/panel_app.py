import numpy as np

from vivarium.simulator.rest_api import SimulatorRestClient
from vivarium.simulator.config import PopulationConfig

import panel as pn
import json
import param

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, HoverTool, Range1d
from bokeh.layouts import layout
from bokeh.events import ButtonClick




def normal(array):
    normals = np.zeros((array.shape[0], map_dim))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals

pn.extension()

simulator = SimulatorRestClient()

sim_config = simulator.get_sim_config()
agent_config = simulator.get_agent_config()
state = simulator.get_state()

def pull_population_config():
    p = PopulationConfig(**PopulationConfig.param.deserialize_parameters(simulator.get_population_config()))
    return p

def update_simulator_population_config(**kwargs):
    simulator.set_population_config(**population_config.param.values(onlychanged=True))

population_config = pull_population_config()

population_config.param.watch_values(update_simulator_population_config, ['n_agents'])

box_size = sim_config['box_size']
map_dim = sim_config['map_dim']

positions = np.array(state['positions'])

N = positions.shape[0]

x = positions[:, 0]
y = x = positions[:, 1]

radius = agent_config['base_length'] / 2.
colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

thetas = np.array(state['thetas'])
normals = normal(thetas)

orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]

cds = ColumnDataSource(data={'x': x, 'y': y,
                             'ox': orientation_lines_x, 'oy': orientation_lines_y,
                             'r': np.ones(N) * radius,
                             'fc': colors
                             }
                       )

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None, mode="vline")
p.add_tools(hover)
p.x_range = Range1d(0, box_size)
p.y_range = Range1d(0, box_size)

orientations = p.multi_line('ox', 'oy', source =cds, color='black', line_width=1)
r = p.circle('x', 'y', radius='r',
             fill_color='fc', fill_alpha=0.6, line_color=None,
             hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=cds)


button = Button(name="Start" if simulator.is_started() else "Stop")


def callback(event):
    if simulator.is_started():
        button.name = "Stop"
        simulator.stop()
    else:
        button.name = "Start"
        simulator.start()


button.on_event(ButtonClick, callback)

draw_tool = PointDrawTool(renderers=[r])
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

#lo = layout([[button], [p, population_config]])
#bk_pane = pn.pane.Bokeh()
#bk_pane.servable()

row = pn.Row(p, population_config)
row.servable()





def update_plot():

    state = simulator.get_state()

    positions = np.array(state['positions'])

    x = positions[:, 0]
    y = positions[:, 1]

    normals = normal(np.array(state['thetas']))

    orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
    orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]

    cds.data['x'] = x
    cds.data['y'] = y
    cds.data['ox'] = orientation_lines_x
    cds.data['oy'] = orientation_lines_y

def update_config():
    pop = pull_population_config()
    vals = dict(pop.param.values())
    del vals['name']
    population_config.param.update(**vals)


pcb1 = pn.state.add_periodic_callback(update_plot, 100)

pcb2 = pn.state.add_periodic_callback(update_config, 2000)


