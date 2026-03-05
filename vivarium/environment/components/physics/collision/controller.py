from .....controllers.controller import Controller


class CollisionController(Controller):

    def __getattr__(self, attr):
        return getattr(self._remote.state, f'{self._name}_state').__getattr__(attr)
    
    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            getattr(self._remote.state, f'{self._name}_state').__setattr__(attr, value)
            