import logging

import hydra
import omegaconf

lg = logging.getLogger(__name__)


class Component:
    def __init__(self, name, precedence):
        self.name = name
        self.precedence = precedence
        self.is_entity_component = False  # No longer needed?

    @classmethod
    def from_config(cls, config, **kwargs):
        kwargs.update(cls.get_kwargs(config))
        return cls(**kwargs)

    def to_config(self, state):
        return omegaconf.OmegaConf.create(
            {
                '_target_': f"{self.__class__.__module__}.{self.__class__.__name__}",
                'precedence': self.precedence,
                'name': self.name
            }
        )

    @staticmethod
    def get_kwargs(config):
        kwargs = {}
        for k, v in config.items():
            if k == '_target_' or k == 'client':
                pass
            elif isinstance(v, (dict, omegaconf.dictconfig.DictConfig)) and '_target_' in v:
                kwargs[k] = hydra.utils.instantiate(v)
            else:
                kwargs[k] = v
        return kwargs

    def update_state_cls(self, state_cls):
        return state_cls

    def init_base_entity(self, entity_state):
        return entity_state

    def init_state_fn(self, state, neighbor_manager, key):
        return state

    def get_step_function(self, state, neighbor_manager, key):
        lg.debug('Component {} has no step function'.format(self.name))
        def step_fn(state, neighbors, key):
            return state
        return step_fn
    
    def neighbor_update(self, state, neighbor_manager, key):
        return state
