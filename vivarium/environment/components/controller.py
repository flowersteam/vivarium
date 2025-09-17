from operator import attrgetter

from vivarium.controllers.dataclass_wrapper import ChangeRecorder


class AttributeMapping:
    def __init__(self, jax_attr, ctrl_attr=None, jax_to_ctrl_fn=None, ctrl_to_jax_fn=None):
        self.jax_attr = jax_attr
        self.ctrl_attr = ctrl_attr if ctrl_attr is not None else jax_attr
        self.jax_to_ctrl_fn = jax_to_ctrl_fn if jax_to_ctrl_fn is not None else lambda x: x
        self.ctrl_to_jax_fn = ctrl_to_jax_fn if ctrl_to_jax_fn is not None else lambda x: x
        


class ComponentController:
    def __init__(self, name, state, mapping):
        self._name = name
        self._state = state
        self._mapping = mapping
        self._change_recorder = ChangeRecorder()

    @classmethod
    def from_config(cls, name, client_config, state, mapping={}, notebook_control=False, **controller_kwargs):
        return cls(
            name=name,
            state=state,
            mapping=mapping,
        )
        
    @property
    def name(self):
        return self._name

    def __getattr__(self, attr):
        mapping = self._mapping[attr]
        return mapping.jax_to_ctrl_fn(attrgetter(mapping.jax_attr)(self._state))

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            object.__setattr__(self, attr, value)
        else:
            attr, value = self.to_jax(attr, value)
            attrgetter(attr)(self._change_recorder).store_change(value)
            
    def to_jax(self, attr, value=None):
        if attr in self._mapping:
            mapping = self._mapping[attr]
            attr = mapping.jax_attr
            if value is not None:
                value = mapping.ctrl_to_jax_fn(value)
        return (attr, value) if value is not None else attr

    def set_state(self, state):
        self._state = state

    def fetch_changes(self):
        changes = self._change_recorder.fetch_changes()
        if changes:
            return [{'state': changes}]
        return []

    def step(self, time, catch_errors):
        pass
    