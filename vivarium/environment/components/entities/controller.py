import numpy as np

from vivarium.controllers.utils import RoutineHandler, Logger
from ....controllers.controller import AttributeMapping

class InternalData:
    pass


def is_split_attribute(attr):
    return attr.startswith('left_') or attr.startswith('right_') or attr.startswith('x_') or attr.startswith('y_')


def split(attr):
    prefix, suffix = attr.split('_', 1)
    suffix = suffix + '_center' if suffix == 'position' else suffix
    return suffix, 0 if prefix == 'left' or prefix == 'x' else 1


def create_property(field_name, rigid_body_field):
    @property
    def prop(self):
        if self._is_rigid_body:
            return getattr(getattr(self._state.entity_state, field_name), rigid_body_field)[self._entity_idx]
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    return self._remote.state.entity_state.orientation[self._entity_idx].obj()
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                return getattr(self._remote.state.entity_state, field_name)[self._entity_idx].obj()
            else:
                return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")

    @prop.setter
    def prop(self, value, idx=None):
        if len(idx) == 0:
            idx = self._entity_idx
        else:
            idx = (self._entity_idx, idx)
        if self._is_rigid_body:
            getattr(getattr(self._remote.state.entity_state, field_name), rigid_body_field)[idx] = value
        else:
            if rigid_body_field == 'orientation':
                if field_name == 'position':
                    self._remote.state.entity_state.orientation[idx] = value
                else:
                    return AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
            elif rigid_body_field == 'center':
                getattr(self._remote.state.entity_state, field_name)[idx] = value
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{field_name}'")
    return prop


class EntityWrapper:
    """
    Wraps a State into an interface to manipulate a single entitty (usually from a SimulatorController)
    Modifying attributes of an EntityWrapper will not change the state immediately, but will record changes
    that can be applied to the state later using the `apply_to_state` method. This is useful for batch updates
    during client-server interactions.
    """
    position_center = create_property('position', 'center')
    momentum_center = create_property('momentum', 'center')
    force_center = create_property('force', 'center')
    mass_center = create_property('mass', 'center')

    position_orientation = create_property('position', 'orientation')
    momentum_orientation = create_property('momentum', 'orientation')
    force_orientation = create_property('force', 'orientation')
    mass_orientation = create_property('mass', 'orientation')

    def __init__(self, remote, ent_idx, entity_type):
        object.__setattr__(self, '_remote', remote)
        object.__setattr__(self, '_entity_idx', ent_idx)
        object.__setattr__(self, '_entity_type_idx', remote.state.entity_state.entity_type_idx.obj()[ent_idx].item())
        object.__setattr__(self, '_is_rigid_body', remote.state.entity_state.obj().is_rigid_body())
        object.__setattr__(self, '_entity_type', entity_type)
        object.__setattr__(self, '_entity_fields', 
                           list(remote.state.entity_state.obj().__dataclass_fields__.keys()) + \
                               [
                                   'position_center', 'position_orientation',
                                   'momentum_center', 'momentum_orientation',
                                   'force_center', 'force_orientation',
                                   'mass_center', 'mass_orientation'
                               ]
                        )
        object.__setattr__(self, 'logger', Logger())

    def __getattr__(self, attr):
        if attr in self._entity_fields:
            return getattr(self._remote.state.entity_state.obj(), attr)[self._entity_idx]
        return getattr(getattr(self._remote.state, self._entity_type), attr).obj()[self._remote.state.entity_state.entity_type_idx.obj()[self._entity_idx]]

    def _setitem(self, attr, value, *idx):
        entity_state_idx = self._entity_idx if len(idx) == 0 else (self._entity_idx, *idx)
        x_state_idx = self._remote.state.entity_state.entity_type_idx[self._entity_idx] if len(idx) == 0 else (self._remote.state.entity_state.entity_type_idx[self._entity_idx], *idx)
        if attr in self._entity_fields:
            if attr.endswith('_center') or attr.endswith('_orientation'):
                field_name, rigid_body_field = attr.split('_', 1)
                p = create_property(field_name, rigid_body_field)
                p.fset(self, value, idx)
            else:
                getattr(self._remote.state.entity_state, attr)[entity_state_idx] = value
        else:
            getattr(getattr(self._remote.state, self._entity_type), attr)[x_state_idx] = value

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        self._setitem(attr, value)


def get_entity_parameter_mapping(subtype_labels):
    return {
        '_default_': lambda attr: AttributeMapping(
            attr,
            remote_to_ctrl_fn=lambda x: x.item() if len(x.shape) == 0 else np.array(x),
            ctrl_to_remote_fn=lambda x: np.array(x)
        ),
        'mass': AttributeMapping(
            'mass',
            remote_to_ctrl_fn=lambda x: x[0].item(),
            ctrl_to_remote_fn=lambda x: np.array([x])
        ),
        'exists': AttributeMapping(
            'exists',
            remote_to_ctrl_fn=lambda x: bool(x.item()),
            ctrl_to_remote_fn=lambda x: np.array(int(x))
        ),
        'subtype': AttributeMapping(
            'entity_subtype',
            remote_to_ctrl_fn=lambda x: subtype_labels[x.item()],
            ctrl_to_remote_fn=lambda x: np.array(subtype_labels.index(x), dtype=int)
        )
    }


class EntityController(EntityWrapper):  # TODO: How about merging the class and the superclass? Actually there is a logic (superclass has same attributes as state)
    """Entity class that represents an entity in the simulation"""

    def __init__(self, remote, ent_idx, entity_type, subtype_labels):
        super().__init__(remote, ent_idx, entity_type)
        object.__setattr__(self, 'internal', InternalData())
        object.__setattr__(self, '_subtype_labels', subtype_labels)
        object.__setattr__(self, '_mapping', get_entity_parameter_mapping(subtype_labels))
        object.__setattr__(self, 'routine_handler', RoutineHandler())
        object.__setattr__(self, '_controller_parameters_fields', getattr(remote.controller_parameters, self._entity_type).obj().__class__.__dataclass_fields__.keys())

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if is_split_attribute(item):
            suffix, idx = split(item)
            field = getattr(self, suffix)
            if suffix == 'position' and self._is_rigid_body:
                field = field.center
            return field[idx]
        if item in self._controller_parameters_fields:
            return getattr(getattr(self._remote.controller_parameters, self._entity_type), item)[self._entity_type_idx]
        pm = self._mapping[item] if item in self._mapping else self._mapping['_default_'](item)
        return pm.remote_to_ctrl_fn(super().__getattr__(pm.remote_attr))

    def __setattr__(self, item, val):
        if item in self._controller_parameters_fields:
            getattr(getattr(self._remote.controller_parameters, self._entity_type), item)[self._entity_type_idx] = val
        elif item in self.__dict__:
            super().__setattr__(item, val)
        elif is_split_attribute(item):
            suffix, idx = split(item)
            self._setitem(suffix, val, idx)
            return
        elif item == 'subtype' and val not in self._subtype_labels:
            raise ValueError(f"'{val}' is not a valid subtype. Valid subtypes are: {self._subtype_labels}")
        else:
            pm = self._mapping[item] if item in self._mapping else self._mapping['_default_'](item)
            super().__setattr__(pm.remote_attr, pm.ctrl_to_remote_fn(val))

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the entity

        :param routine_fn: routine_fn
        :param name: routine name, defaults to None
        :param interval: routine execution interval, defaults to 1
        """
        self.routine_handler.attach_routine(routine_fn, name, interval)

    def detach_routine(self, name):
        """Detach a routine from the entity

        :param name: routine name
        """
        self.routine_handler.detach_routine(name)

    def detach_all_routines(self):
        """Detach all routines from the entity"""
        self.routine_handler.detach_all_routines()

    def step(self, time, catch_errors):
        """Execute the entity's routines with their corresponding execution intervals"""
        # Give self object as parameter to the routine function so it executes functions on the entity
        self.routine_handler.routine_step(self, time, catch_errors)

    def print_infos(self):
        # TODO: to fix according to recent refactoring
        """Print the entity's infos

        :return: entity's infos
        """

        info_lines = []
        info_lines.append("Entity Overview:")
        info_lines.append(f"{'-' * 20}")
        info_lines.append(f"Type: {self._entity_type}")
        info_lines.append(f"Subtype: {self._subtype_labels[self.entity_subtype]}")
        info_lines.append(f"Idx: {self._entity_type_idx}")
        info_lines.append(f"Exists: {bool(self.exists)}")
        info_lines.append(
            f"Position: x={self.x_position:.2f}, y={self.y_position:.2f}"
        )
        info_lines.append(f"Diameter: {self.diameter:.2f}")
        info_lines.append(f"Color: {self.color}")
        info_lines.append("")

        return print("\n".join(info_lines))

    def print_routines(self):
        """Print the entity's routines"""
        self.routine_handler.print_routines()


class EntityList:
    def __init__(self, remote, entity_type, entity_type_idx, entity_wrapper_list=None):
        self._remote = remote
        self.entity_type = entity_type
        self._entity_list = entity_wrapper_list or [EntityWrapper(remote, idx, entity_type) for idx, type in enumerate(remote.state.entity_state.entity_type.obj()) if type == entity_type_idx]
    
    def __getitem__(self, idx):
        return self._entity_list[idx]

    def __setitem__(self, idx, value):
        raise NotImplementedError('Setting values directly is not supported.')

    def __iter__(self):
        return iter(self._entity_list)

    def __len__(self):
        return len(self._entity_list)

    def __repr__(self):
        return repr(self._entity_list)


class EntityListController(EntityList):
    def __init__(self, entity_type, remote, 
                 subtype_labels=None,  # TODO: not used yet but should be to access/change it from the VivariumController
                 controller_cls=None,
                 ):
        controller_cls = EntityController if controller_cls is None else controller_cls
        self.subtype_labels = subtype_labels
        self.name = entity_type

        etype_int = getattr(remote.state, entity_type).entity_type
        super().__init__(
            remote=remote, entity_type=entity_type, entity_type_idx=etype_int,
            entity_wrapper_list=[
                controller_cls(remote, idx, entity_type, subtype_labels)
                for idx, type in enumerate(remote.state.entity_state.entity_type.obj())
                if type == etype_int]            
        )
        
    @classmethod
    def from_config(cls, name, remote):
        return cls(
            entity_type=name,
            remote=remote,
            subtype_labels=remote.controller_parameters.simulator.subtype_labels.obj()
        )
        
    def step(self, time, catch_errors):
        # TODO : Add a check to ensure that the entity exists
        for entity in self._entity_list:
            entity.step(time, catch_errors=catch_errors)
