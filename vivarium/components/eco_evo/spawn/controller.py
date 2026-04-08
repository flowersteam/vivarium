import numpy as np

from vivarium.controllers.controller import Controller
from vivarium.utils.dataclass_wrapper import Remote


class SingleSpawnController:
    def __init__(self, remote, idx, global_controller_name, subtype_labels):
        self._remote = remote
        self._idx = idx
        self._global_controller_name = global_controller_name
        self._subtype_labels = subtype_labels

    def __getattr__(self, attr):
        if attr.startswith('_'):
            return object.__getattribute__(self, attr)
        value = getattr(self._remote.state, f'{self._global_controller_name}_state').__getattr__(attr)[self._idx]
        if attr == 'subtype':
            return self._subtype_labels[int(value)]
        # Normalize: value may be a Remote proxy (wrapping JAX/numpy), a bare
        # JAX array, or a plain Python type depending on the access path.
        if isinstance(value, Remote):
            value = value.obj()
        value = np.asarray(value)
        if attr in ('position_range', 'orientation_range'):
            return tuple(value.tolist())
        return value.item()

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            if attr == 'subtype':
                value = np.array(self._subtype_labels.index(value), dtype=int)
            self._remote.state.__getattr__(f'{self._global_controller_name}_state').__getattr__(attr)[self._idx] = value


class SpawnController(Controller):
    def __init__(self, name, remote, mapping=None):
        self._subtype_labels = remote.controller_parameters.simulator.subtype_labels.obj()

        # Get spawn names from controller_parameters; fall back to ['default'] for legacy scenes
        try:
            self._spawn_names = remote.controller_parameters.spawn.names.obj()
        except AttributeError:
            self._spawn_names = ['default']

        self._single_spawn_controllers = {
            s_name: SingleSpawnController(remote, idx, name, self._subtype_labels)
            for idx, s_name in enumerate(self._spawn_names)
        }

        super().__init__(name, remote, mapping=mapping or {})

    def __getattr__(self, attr):
        if self.to_deal_with(attr):
            if attr in self._spawn_names:
                return self._single_spawn_controllers[attr]
            elif len(self._spawn_names) == 1:
                # Single config: allow direct attribute access
                return self._single_spawn_controllers[self._spawn_names[0]].__getattr__(attr)
            else:
                raise AttributeError(
                    f"Spawn '{attr}' not found. Access a specific spawn rule among: {list(self._spawn_names)}"
                )
        else:
            return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if self.to_deal_with(attr):
            if len(self._spawn_names) == 1:
                # Single config: allow direct attribute access
                self._single_spawn_controllers[self._spawn_names[0]].__setattr__(attr, value)
            else:
                raise AttributeError(
                    f"Spawn '{attr}' not found. Access a specific spawn rule among: {list(self._spawn_names)}"
                )
        else:
            super().__setattr__(attr, value)
