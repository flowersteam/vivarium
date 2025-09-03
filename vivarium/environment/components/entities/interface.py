import param
import panel as pn
from enum import Enum
from dataclasses import asdict
from panel.layout import Column
from bokeh.plotting import figure
from contextlib import contextmanager
from bokeh.models import BooleanFilter, CDSView, ColumnDataSource

import numpy as np

from vivarium.controllers.panel_controller import ParameterizedData


class Shape(Enum):
    CIRCLE = 0
    SQUARE = 1


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


class Selected(param.Parameterized):
    """Class to store the selected entities in the interface"""

    selection = param.ListSelector([0], objects=[0])

    def __len__(self):
        return len(self.selection)


class ParamEntity(ParameterizedData):
    x_position = param.Number()
    y_position = param.Number()
    orientation = param.Number()
    mass = param.Number()
    diameter = param.Number()
    friction = param.Number()
    exists = param.Boolean()
    color = param.Color()
    shape = param.String()
    visible = param.Boolean()
    hide_non_existing = param.Boolean()

    def __init__(self, entities, subtype_labels, **params):

        self.subtype_labels = subtype_labels
        self.subtype_label_list = [self.subtype_labels[i] for i in sorted(self.subtype_labels)]
        self.param.add_parameter('subtype', param.Selector(objects=self.subtype_label_list))

        super().__init__(entities,
                         **params)
        
        self.selection = [0]

    @property
    def selected_entity_data(self):
        return self.data[self.selection[0]]


class EntityRenderer:
    def __init__(
        self,
        entities,
        selected_param_entity,
        selected, etype, state,
        shape,
        line_width=1.0,
    ):
        self.entities = entities
        self.selected_param_entity = selected_param_entity
        self.selected = selected
        self.etype = etype
        
        self.panel_visibility_parameters = [p for p in self.selected_param_entity.direct_mapping_parameters if p.startswith('visible')]
        
        # TODO: for now only the shape of the first entity is considered
        self.shape = getattr(Shape, shape[0].upper()) if isinstance(shape[0], str) else shape[0]
        
        self.line_width = line_width
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        self.cds.on_change("data", self.drag_cb)
        self.cds_view = self.create_cds_view()
        selected.param.watch(
            self.update_selected_plot, ["selection"], onlychanged=True, precedence=0
        )

        self.selected_param_entity.param.watch(self.update_cds_view,
                                               self.panel_visibility_parameters,
                                               onlychanged=True)
        self.selected_param_entity.param.watch(self.apply_visible_filter,
                                               ["exists", "hide_non_existing"],
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
            for attr in self.panel_visibility_parameters
        }

    def update_cds_view(self, event):
        """Updates the view of the ColumnDataSource if the visibility of an entity changes

        :param event: The event containing the changed value
        """
        n = event.name
        for attr in [n] if n != "visible" else self.panel_visibility_parameters:
            f = [getattr(e, attr) and e.visible for e in self.entities]
            self.cds_view[attr].filter = BooleanFilter(f)

    def update_selected_plot(self, event):
        """Updates the selected entities in the plot

        :param event: The event containing the new selected entities
        """
        self.cds.selected.indices = event.new


    def apply_visible_filter(self, *args, **kwargs):
        f = [e.visible for e in self.entities]
        for attr in self.panel_visibility_parameters:
            self.cds_view[attr].filter = BooleanFilter(f)

    def update(self):
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


class EntityInterface:
    
    param_cls = ParamEntity
    renderer_cls = EntityRenderer
    
    def __init__(self, controller, state, subtype_labels, panel_cls=Column):
        
        self.parameters = self.param_cls(controller, subtype_labels, **asdict(controller.controller_parameters[0]))
        self.parameters.update_from_server = True
        
        self.selected = Selected()
        self.selected.param.selection.objects = state.entity_type_idx(controller.entity_type).tolist() 
        
        self.renderer = self.renderer_cls(
                entities = controller._entity_list,
                selected_param_entity=self.parameters,
                selected=self.selected,
                etype=controller.entity_type,
                state=state,
                shape=controller.controller_parameters.shape,
            )

        self.widget = panel_cls(
                pn.pane.Markdown(f"### {controller.entity_type}", align="center"),
                self.selected,
                pn.panel(
                    self.parameters,
                    name="State configuration",
                ),
                visible=True,
                sizing_mode="scale_height",
                scroll=True,
                name=controller.entity_type,            
            )
        

        self.selected.param.watch(
            self.pull_selected_entities,
            ["selection"],
            onlychanged=True,
            precedence=1,
        )

    def pull_selected_entities(self, *events):
        """Pull the selected configurations"""
        self.parameters.selection = self.selected.selection
        self.parameters.update_from_server = True



