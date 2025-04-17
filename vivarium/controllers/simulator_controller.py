import numpy as np

from vivarium.environments.braitenberg.behaviors import Behaviors, behavior_to_params
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.controllers.dataclass_wrapper import (
    EntityList, EntityWrapper, SimulatorParametersWrapper
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
        cur_params = np.array(self.behavior_params)
        cur_params[slot_idx] = behavior_to_params(behavior)
        self.behavior_params = cur_params


class ControllerObject(ControllerEntity):
    pass
    

def create_entity_lists(state, etype_to_class, etype_to_idx, etype_to_kwargs):

    entity_lists = {
        etype: EntityList(
            state=state, entity_type=etype, entity_type_idx=etype_to_idx[etype],
            entity_wrapper_list=[
                eclass(state, idx, etype, 
                       **{attr: val[int(state.entity_state.entity_type_idx[idx])]
                          for attr, val in etype_to_kwargs[etype].items()}
                       ) 
                for idx, type in enumerate(state.entity_state.entity_type)
                if type == etype_to_idx[etype]]
        )
        for etype, eclass in etype_to_class.items()
    }
    return entity_lists

def entity_type_property(entity_type):
    @property
    def prop(self):
        return self.entity_lists[entity_type]
    return prop

class SimulatorController:
    config_field = 'simulator_controller'

    def __init__(self, client=None):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.simulator_parameters = self.client.get_simulator_parameters()
        self.scene_config = SceneConfiguration(self.client.scene_name)
        self.subtype_labels = self.scene_config.subtype_labels
        self.create_entity_lists()
        for etype in self.scene_config.entity_types:
            setattr(self, etype, self.entity_lists[etype])
        self.create_simulator_parameters_wrapper()
        
    def create_entity_lists(self):
        etype_to_class = {etype: getattr(config, self.config_field).cls for etype, config in self.scene_config.entity_type_client_configs.items()}
        etype_to_idx = {etype: config.idx for etype, config in self.scene_config.entity_type_configs.items()}
        etype_to_kwargs = {etype: getattr(config, self.config_field).kwargs for etype, config in self.scene_config.entity_type_client_configs.items()}
        self.entity_lists = create_entity_lists(self.state, etype_to_class, etype_to_idx, etype_to_kwargs)

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
        change = self.simulator_parameters.fetch_changes()
        if change:
            changes.extend([change])
        return changes

    def apply_changes(self):
        changes = self.fetch_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)
