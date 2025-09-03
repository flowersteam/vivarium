import param
import panel as pn
from panel.layout import Column

from vivarium.controllers.panel_controller import ParameterizedData


class CollisionParam(ParameterizedData):
    epsilon = param.Number()
    alpha = param.Number()

    def __init__(self, controller, **params):
        super().__init__(data=controller, **params)


class CollisionInterface:
    def __init__(self, controller, state, subtype_labels, panel_cls=Column):
        
        self.parameters = CollisionParam(controller=controller)
        
        self.parameters.update_from_server = True
        
        self.widget = panel_cls(
                pn.pane.Markdown(f"### {controller.name}", align="center"),
                pn.panel(
                    self.parameters,
                    name="State configuration",
                ),
                visible=True,
                sizing_mode="scale_height",
                scroll=True,
                name=controller.name,            
            )
        
        self.renderer = None
        