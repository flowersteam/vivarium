import param
from panel.layout import Column

from vivarium.environment.components.interface import Interface, Renderer
from vivarium.interface.parameterized import ParameterizedData

class WallRenderer(Renderer):
    
    def __init__(self, entity_type, state):
        
        self.entity_type = entity_type
        
        super().__init__(state)
    
    def get_cds_data(self, state):
        x = getattr(state, self.entity_type).coordinates[:, :, 0].tolist()
        y = getattr(state, self.entity_type).coordinates[:, :, 1].tolist()

        data = dict(x=x, y=y)
        return data
    
    def plot(self, fig):
        
        src = {"source": self.cds}
        
        fig.multi_line("x", "y", color="white", **src)
        
        return fig


class WallParam(ParameterizedData):
    epsilon = param.Number()
    alpha = param.Number()

    def __init__(self, controller, **params):
        super().__init__(controller=controller, **params)
        

class WallInterface(Interface):
    def __init__(self, controller, state, panel_cls=Column):
        parameters = WallParam(controller=controller)
        
        renderer = WallRenderer(controller.name, state)
        
        super().__init__(controller, parameters, panel_cls=panel_cls, renderer=renderer)
