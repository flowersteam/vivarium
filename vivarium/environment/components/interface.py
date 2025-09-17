import panel as pn
from panel import Column
from bokeh.models import ColumnDataSource


class Renderer:
    
    def __init__(self, state, use_point_draw_tool=False):
        
        self.cds = ColumnDataSource(data=self.get_cds_data(state))

        self.use_point_draw_tool = use_point_draw_tool
    
    def get_cds_data(self, state):
        """Update the ColumnDataSource with the new data

        :param state: The state coming from the server
        :return: Data dictionary for the ColumnDataSource
        """
        raise NotImplementedError
    
    def update_cds(self, state):
        """Updates the ColumnDataSource with new data from server

        :param state: The state coming from the server
        """
        self.cds.data.update(self.get_cds_data(state))
    
    def plot(self, fig):

        raise NotImplementedError

    def update(self):
        pass



class Interface:
    def __init__(self, controller, parameters, panel_cls=Column, renderer=None, default_widget=True):
        self.parameters = parameters
        self.renderer = renderer
        self.controller = controller
        self.panel_cls = panel_cls
        if default_widget:
            self.widget = self.default_widget()

    def default_widget(self):
        self.parameters.update_from_server = True
        return self.panel_cls(
                pn.pane.Markdown(f"### {self.controller.name}", align="center"),
                pn.panel(
                    self.parameters,
                    widgets={param_name: {'width': 100, 'min_width': 80, 'max_width': 140} for param_name in self.parameters.param_names()},
                    name="Parameters",
                ),
                visible=True,
                sizing_mode="scale_height",
                scroll=True,
                name=self.controller.name,            
            )