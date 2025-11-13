from vivarium.controllers.controller import Controller


class SimulatorController(Controller):

    def __getattr__(self, attr):
        return getattr(self._remote.controller_parameters.simulator, attr)
    
    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            setattr(self._remote.controller_parameters.simulator, attr, value)
            