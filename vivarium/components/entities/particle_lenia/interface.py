import param
from vivarium.components.entities.particle_lenia.creatures import CREATURES
from vivarium.components.entities.interface import ParamEntity, EntityInterface

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
                setattr(self.controller[i], attr, creature[attr])


class ParticleLeniaInterface(EntityInterface):

    param_cls = ParamParticleLenia
    