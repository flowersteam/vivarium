import numpy as np
import param
from vivarium.controllers.panel_controller import ParamEntity
from vivarium.environment.components.entities.controller import EntityController
from vivarium.environment.components.entities.particle_lenia.creatures import CREATURES

class ParamParticleLenia(ParamEntity):
    mu_k = param.Number()
    sigma_k = param.Number()
    w_k = param.Number()
    mu_g = param.Number()
    sigma_g = param.Number()
    c_rep = param.Number()

    def __init__(self, entities, subtype_labels, **params):
        super().__init__(entities, subtype_labels, **params)
        self.param.add_parameter('creature', param.Selector(objects=list(CREATURES.keys())))
        self.param.watch(self.update_creature, 'creature', onlychanged=True)

    def update_creature(self, event):
        """Update the parameters based on the selected creature."""
        creature = CREATURES[event.new]
        for i in self.selection:
            for attr in ['mu_k', 'sigma_k', 'w_k', 'mu_g', 'sigma_g', 'c_rep']:
                setattr(self.data[i], attr, creature[attr])


class ParticleLeniaController(EntityController):
    def __init__(self, entity_type, 
                 param_cls=ParamParticleLenia,
                 **kwargs):
        super().__init__(
            entity_type=entity_type,
            param_cls=param_cls,
            **kwargs
        )
