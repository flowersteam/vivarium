import hydra

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.utils.scene_configs import extend_kwargs
from vivarium.controllers.dataclass_wrapper import (
    ChangeRecorder, EntityWrapper, SimulatorParametersWrapper
)


class InternalData:
    pass


def is_split_attribute(attr):
    return attr.startswith('left_') or attr.startswith('right_') or attr.startswith('x_') or attr.startswith('y_')


def split(attr):
    prefix, suffix = attr.split('_', 1)
    suffix = suffix + '_center' if suffix == 'position' else suffix
    return suffix, 0 if prefix == 'left' or prefix == 'x' else 1


class ControllerEntity(EntityWrapper):
    """Entity class that represents an entity in the simulation"""

    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type)
        object.__setattr__(self, 'controller_parameters', controller_parameters)
        object.__setattr__(self, '_controller_change_recorder', ChangeRecorder())
        object.__setattr__(self, 'internal', InternalData())

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if is_split_attribute(item):
            suffix, idx = split(item)
            field = getattr(self, suffix)
            if suffix == 'position' and self._is_rigid_body:
                field = field.center
            return field[idx]
        if item in self.controller_parameters.__class__.__dataclass_fields__:
            return getattr(self.controller_parameters, item)
        return super().__getattr__(item)

    def __setattr__(self, item, val):
        if item in self.controller_parameters.__class__.__dataclass_fields__:
            getattr(self._controller_change_recorder, item)[self._entity_type_idx] = val
            object.__setattr__(self.controller_parameters, item, val)
            return
        if item in self.__dict__:
            super().__setattr__(item, val)
        elif is_split_attribute(item):
            suffix, idx = split(item)
            self._setitem(suffix, val, idx)
            return
        else:
            super().__setattr__(item, val)

class SimulatorController:

    def __init__(self, client=None, subtypes=[], **controllers):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.simulator_parameters = self.client.get_simulator_parameters()
        self.subtype_labels = {i: label for i, label in enumerate(subtypes)}
        
        self.controllers = controllers
        
        self.entity_lists = self.create_entity_list()

        for etype, elist in self.entity_lists.items():
            setattr(self, etype, elist)
        self.create_simulator_parameters_wrapper()

    @classmethod
    def from_config(cls, config, client=None):
        controllers = {}
        for etype, e_config in config.client_list.items():
            e_cls = hydra.utils.get_class(e_config.cls)
            kwargs = extend_kwargs(e_config.controller_kwargs, e_config.n_max)
            controllers[etype] = e_cls(etype, **kwargs)
        return cls(
            subtypes=config.subtypes,
            client=client,
            **controllers
        )

    def create_entity_list(self):
        return {etype: c.controller(self.state) for etype, c in self.controllers.items()}

    def create_simulator_parameters_wrapper(self):
        self.simulator_parameters = SimulatorParametersWrapper(self.simulator_parameters)

    def start(self):
        """Start the simulator."""
        self.client.start()

    def stop(self):
        """Stop the simulator."""
        self.client.stop()

    def is_started(self):
        """Check if the simulator is started."""
        return self.client.is_started()

    def step(self):
        changes = self.fetch_changes()
        self.state = self.client.step(changes)
        self.update_entity_lists()

    def update_entity_lists(self, state=None):
        """Update the entity lists."""
        state = state or self.state
        for _, ent_list in self.entity_lists.items():
            ent_list.set_state(state)

    def update_state(self):
        """Update the state from server to client."""
        self.state = self.client.get_state()
        self.update_entity_lists()
        return self.state

    def fetch_changes(self):
        changes = []
        for etype, elist in self.entity_lists.items():
            change = elist.fetch_changes()
            change = [{'state': c} for c in change]
            changes.extend(change)
            for e in elist:
                change = e._controller_change_recorder.fetch_changes()
                if change:
                    changes.extend([{'controller_parameters': {etype: change}}])
        change = self.simulator_parameters.fetch_changes()
        if change:
            changes.extend([change])
        return changes

    def apply_changes(self):
        changes = self.fetch_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)
