import numpy as np
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.dataclass_wrapper import (
    EntityList, EntityWrapper, SimulatorStateWrapper
)
from vivarium.simulator.simulator_states import EntityType
from vivarium.utils.converters import string_to_rgb_array
from vivarium.environments.braitenberg.behaviors import Behaviors, behavior_to_params


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

    def __init__(self, state, ent_idx, entity_type):
        super().__init__(state, ent_idx, entity_type)
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
        return super().__getattr__(item)
    
    def __setattr__(self, item, val):
        if item in self.__dict__:
            super().__setattr__(item, val)
        elif is_split_attribute(item):
            suffix, idx = split(item)
            self._setitem(suffix, val, idx)
            return
        else:
            if item == 'color' and isinstance(val, str):
                val = string_to_rgb_array(val)
            super().__setattr__(item, val)

class ControllerAgent(ControllerEntity):

    def set_behavior(self, slot_idx, behavior, sensed):
        assert slot_idx < self.behavior.shape[0], 'Behavior index out of bounds'
        if isinstance(behavior, Behaviors):
            behavior = behavior.value
        cur_behaviors = np.array(self.behavior)
        cur_behaviors[slot_idx] = behavior
        self.behavior = cur_behaviors
        cur_sensed = np.array(self.sensed)
        cur_sensed[slot_idx] = [int(i in sensed) for i in range(len(cur_sensed[slot_idx]))]
        self.sensed = cur_sensed
        cur_params = np.array(self.params)
        cur_params[slot_idx] = behavior_to_params(behavior)
        self.params = cur_params

class ControllerObject(ControllerEntity):
    pass
    

def create_entity_lists(state, etype_to_class):
    entity_lists = {
        etype: EntityList(
            state=state, entity_type=etype,
            entity_wrapper_list=[
                eclass(state, idx, etype) 
                for idx, type in enumerate(state.entity_state.entity_type) if type == etype.value]
        )
        for etype, eclass in etype_to_class.items()
    }
    return entity_lists


class SimulatorController:
    def __init__(self, client=None):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.create_entity_lists()
        self.create_simulator_state()
        
    def create_entity_lists(self):
        self.entity_lists = create_entity_lists(self.state, 
                                                {EntityType.AGENT: ControllerAgent,
                                                 EntityType.OBJECT: ControllerObject})

    def create_simulator_state(self):
        self.simulator_state = SimulatorStateWrapper(self.state)

    @property
    def agents(self):
        return self.entity_lists[EntityType.AGENT]
    
    @property
    def objects(self):
        return self.entity_lists[EntityType.OBJECT]

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

    def update_simulator_state(self, state=None):
        """Update the simulator state."""
        state = state or self.state
        self.simulator_state.set_state(self.state)

    def update_state(self):
        """Update the state of the simulator."""
        self.state = self.client.get_state()
        self.update_entity_lists()
        self.update_simulator_state()
        return self.state

    def fetch_changes(self):
        changes = []
        for etype, elist in self.entity_lists.items():
            changes.extend(elist.fetch_changes())
        changes.extend([self.simulator_state.fetch_changes()])
        return changes

    def apply_changes(self):
        changes = self.fetch_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)

    def get_subtype_labels(self):
        return self.client.get_subtype_labels()
