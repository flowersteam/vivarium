from vivarium.controllers.dataclass_wrapper import EntityList, create_dataclass_from_dict
from vivarium.controllers.notebook_controller import NotebookControllerEntity
from vivarium.controllers.simulator_controller import ControllerEntity


def entity_list(controller_cls, state, entity_type, entity_type_int, controller_parameters):
    return EntityList(
        state=state, entity_type=entity_type, entity_type_idx=entity_type_int,
        entity_wrapper_list=[
            controller_cls(state, idx, entity_type,
                            controller_parameters=controller_parameters[int(state.entity_state.entity_type_idx[idx])])
            for idx, type in enumerate(state.entity_state.entity_type)
            if type == entity_type_int]
    )

class EntityController:
    def __init__(self, entity_type, state, 
                 subtype_labels,  # TODO: not used yet but should be to access/change it from the SimulatorController
                 controller_cls=ControllerEntity, 
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
