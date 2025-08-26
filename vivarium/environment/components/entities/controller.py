from vivarium.controllers.utils import RoutineHandler
from vivarium.controllers.dataclass_wrapper import ChangeRecorder, EntityList, EntityWrapper, create_dataclass_from_dict


class InternalData:
    pass


def is_split_attribute(attr):
    return attr.startswith('left_') or attr.startswith('right_') or attr.startswith('x_') or attr.startswith('y_')


def split(attr):
    prefix, suffix = attr.split('_', 1)
    suffix = suffix + '_center' if suffix == 'position' else suffix
    return suffix, 0 if prefix == 'left' or prefix == 'x' else 1


class EntityController(EntityWrapper):
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


class NotebookControllerEntity(EntityController):
    """Entity class that represents an entity in the simulation"""

    def __init__(self, state, ent_idx, entity_type, controller_parameters):
        super().__init__(state, ent_idx, entity_type, controller_parameters)
        object.__setattr__(self, 'routine_handler', RoutineHandler())
        object.__setattr__(self, 'controller_parameters', controller_parameters)

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
        dict_infos = self.config.to_dict()

        info_lines = []
        info_lines.append("Entity Overview:")
        info_lines.append(f"{'-' * 20}")
        info_lines.append(f"Type: {self.etype.name}")
        info_lines.append(f"Subtype: {self.subtype_label}")
        info_lines.append(f"Idx: {self.idx}")
        info_lines.append(f"Exists: {self.exists}")
        info_lines.append(
            f"Position: x={dict_infos['x_position']:.2f}, y={dict_infos['y_position']:.2f}"
        )
        info_lines.append(f"Diameter: {self.diameter:.2f}")
        info_lines.append(f"Color: {self.color}")
        info_lines.append("")

        return print("\n".join(info_lines))

    def print_routines(self):
        """Print the entity's routines"""
        self.routine_handler.print_routines()


def entity_list(controller_cls, state, entity_type, entity_type_int, controller_parameters):
    return EntityList(
        state=state, entity_type=entity_type, entity_type_idx=entity_type_int,
        entity_wrapper_list=[
            controller_cls(state, idx, entity_type,
                            controller_parameters=controller_parameters[int(state.entity_state.entity_type_idx[idx])])
            for idx, type in enumerate(state.entity_state.entity_type)
            if type == entity_type_int]
    )
    

class EntityListController:
    def __init__(self, entity_type, state, 
                 subtype_labels,  # TODO: not used yet but should be to access/change it from the SimulatorController
                 controller_cls=EntityController, 
                 notebook_controller_cls=NotebookControllerEntity,
                 **kwargs
                 ):
        self.controller_cls = controller_cls
        self.notebook_controller_cls = notebook_controller_cls
        
        self.entity_type = entity_type
        self.controller_parameters = create_dataclass_from_dict(
            'ControllerParameters',
            kwargs)
        self.controller = self.create_controller(state)  

    def create_controller(self, state, controller_cls=None):
        controller_cls = controller_cls or self.controller_cls
        etype_int = getattr(state, self.entity_type).entity_type
        return entity_list(
            controller_cls=controller_cls,
            state=state,
            entity_type=self.entity_type,
            entity_type_int=etype_int,
            controller_parameters=self.controller_parameters
        )
