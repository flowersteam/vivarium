from ...controller import ComponentController


class SpawnController(ComponentController):
    def __init__(self, name, state, mapping=None):
        super().__init__(name, getattr(state, f'{name}_state'), mapping=mapping, path=('state', f'{name}_state'))
