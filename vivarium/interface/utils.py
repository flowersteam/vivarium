from vivarium.environment.components.entities.braitenberg.interface import ParamAgent
from vivarium.environment.components.entities.particle_lenia.interface import ParamParticleLenia


param_fields_to_delete = {
     ParamAgent: lambda field: field.startswith('sensed_') or field.startswith('behavior_'),
     ParamParticleLenia: lambda field: field == 'creature'
}

def cleanup_parameterized_class():
    """
    Remove the dynamically added parameters from the Param classes
    as they might be remnants from previous instaciations
    """
    for param_cls, condition in param_fields_to_delete.items():
        to_del = []
        for field_name in param_cls.__dict__.keys():
            if condition(field_name):
                to_del.append(field_name)
        for field_name in to_del:
            delattr(param_cls, field_name)
            del param_cls._param__parameters._cls_parameters[field_name]