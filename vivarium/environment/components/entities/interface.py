import jax.numpy as jnp
import param
from enum import Enum
from contextlib import contextmanager
import panel as pn
from panel.layout import Column

from bokeh.models import BooleanFilter, CDSView, ColumnDataSource
from bokeh.plotting import figure
import numpy as np

from vivarium.controllers.panel_controller import ParameterMapping, ParameterizedData

class Shape(Enum):
    CIRCLE = 0
    SQUARE = 1


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


entity_parameter_mapping = {
    'orientation': ParameterMapping('position_orientation'),
    'mass': ParameterMapping(
        'mass_center',
        jax_to_param_fn=lambda x: x[0].item(),
        param_to_jax_fn=lambda x: jnp.array([x])
    ),
    'exists': ParameterMapping(
        'exists',
        jax_to_param_fn=lambda x: bool(x.item()),
        param_to_jax_fn=lambda x: jnp.array(int(x))
    ),
}


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
    visible = param.Boolean()
    hide_non_existing = param.Boolean(True)

    def __init__(self, entities, subtype_labels, parameter_mapping={}, panel_parameters=[], **params):

        self.subtype_labels = subtype_labels
        self.subtype_label_list = [self.subtype_labels[i] for i in sorted(self.subtype_labels)]
        self.param.add_parameter('subtype', param.Selector(objects=self.subtype_label_list))
        parameter_mapping.update(entity_parameter_mapping)
        parameter_mapping['subtype'] = ParameterMapping(
            'entity_subtype',
            jax_to_param_fn=lambda x: self.subtype_label_list[x.item()],
            param_to_jax_fn=lambda x: jnp.array(self.subtype_label_list.index(x), dtype=int)
        )
        super().__init__(entities,
                         parameter_mapping=parameter_mapping,
                         panel_parameters=panel_parameters + ['visible', 'color', 'hide_non_existing'],
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
        # param_simulator_state,
        selected, etype, state,
        shape,
        line_width=1.0,
    ):
        self.entities = entities
        self.selected_param_entity = selected_param_entity
        # self.param_simulator_state = param_simulator_state
        self.selected = selected
        self.etype = etype
        
        # TODO: for now only the shape of the first entity is considered
        self.shape = getattr(Shape, shape[0].upper()) if isinstance(shape[0], str) else shape[0]
        
        self.line_width = line_width
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        self.cds.on_change("data", self.drag_cb)
        self.cds_view = self.create_cds_view()
        self.selected_param_entity.param.watch(
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
        if self.selected_param_entity.hide_non_existing:
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
        if not self.selected_param_entity.hide_non_existing:
            return
        self.selected_param_entity.visible = event.new


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
        
        self.parameters = self.param_cls(controller, subtype_labels)
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



