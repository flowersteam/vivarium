import os 
import random
import logging
from math import pi
from collections.abc import Iterable

from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
import hydra

from vivarium.environments.state import create_state_cls
from vivarium.controllers.dataclass_wrapper import create_dataclass_from_dict
from vivarium.utils.converters import import_class
from vivarium.simulator import Simulator


abs_config_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../conf"))
config_dir_path = os.path.relpath(abs_config_dir_path, start=os.path.dirname(__file__))


def generate_random_positions(n, position_range, seed=None):
    """
    Generate random positions within a given range
    :param n: number of positions to generate
    :param position_range: range of positions (x_min, x_max, y_min, y_max)
    :param seed: random seed
    :return: list of random positions
    """
    x_min, x_max, y_min, y_max = position_range
    rng = random.Random(seed)
    return [[rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)] for _ in range(n)]


def generate_random_orientations(n, seed=None):
    # Generate random orientations
    rng = random.Random(seed)
    return [rng.uniform(0, 2 * pi) for _ in range(n)]


def load_scene_config(scene_name: str, seed=None) -> DictConfig:
    """Load a specific scene configuration

    :param scene_name: scene name of yaml file
    :return: scene configuration
    """
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    with hydra.initialize(config_path=config_dir_path, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"scene={scene_name}"])
        logging.basicConfig(level=cfg.log_level)
        return cfg.scene


def extend_kwargs(kwargs, n):
    """Extend kwargs to n items"""
    for attr, val in kwargs.items():
        if isinstance(val, Iterable) and '_all_values_' in val:
            kwargs[attr] = [val['_all_values_']] * n
    return kwargs


def get_n_max(config_node):
    return config_node.n_max if 'n_max' in config_node else config_node.n_exists


class SceneConfiguration:
    
    def __init__(self, config, seed=None):

        if type(config) is str:
            self.config = load_scene_config(config)
        else:
            self.config = config
        self.scene_name = self.config.scene_name
        self.seed = seed
        
    def compute_parameters(self, name, params):
        assert 'n_exists' in params or 'n_max' in params, f"Either n_max ot n_exists has to be defined"
        n_max = get_n_max(params)
        n_exists = params.n_exists if 'n_exists' in params else n_max
        OmegaConf.update(self.config.environment, 'components[' + name + ']', {'n_max': n_max}, force_add=True)
        OmegaConf.update(self.config.environment, 'components[' + name + ']', {'n_exists': n_exists}, force_add=True)
        n = params.n_max
        if params.position == '_random_':
            # Generate random positions if not provided
            pos_range = [0, self.config.environment.kwargs.box_size,
                            0, self.config.environment.kwargs.box_size]
            params.position = generate_random_positions(n, pos_range, self.seed)
        elif '_range_' in params.position:
            # Generate random positions within a specified range
            params.position = generate_random_positions(n, params.position['_range_'], self.seed)
        if params.orientation == '_random_':
            # Generate random orientations if not provided
            params.orientation = generate_random_orientations(n, self.seed)

        params = extend_kwargs(params, n)

        return params

    def create_state_cls(self):
        base_state_cls = hydra.utils.get_class(self.config.environment.kwargs.base_state_cls)
        update_fns = [f.update_state_cls for f in self.create_component_factories()]
        return create_state_cls(
            base_state_cls=base_state_cls,
            update_fns=update_fns
        )

    def create_component_factories(self):
        component_factories = []
        for k, v in self.config.environment.components.items():
            f = hydra.utils.get_class(v._target_).from_config(name=k, scene_config=self, config_node=v)
            component_factories.append(f)
        return component_factories

    def create_environment(self):
        env_cls = import_class(self.config.environment.cls)
        kwargs = {}
        for k, v in self.config.environment.kwargs.items():
            if k == 'base_state_cls':
                kwargs['base_state_cls'] = hydra.utils.get_class(v)
            else:
                kwargs[k] = v
        env = env_cls.init_neighbor_manager(**kwargs, factories=self.create_component_factories())
        return env
    
    def create_simulator(self, env=None):
        env = env or self.create_environment()
        cp = self.create_controller_parameters()
        return Simulator(env=env, scene_name=self.scene_name, 
                         controller_parameters=cp,
                         freq=self.config.simulator.kwargs.freq)

    def create_controller_parameters(self):
        kwargs = {}
        for etype, config in self.config.client.items():
            n_max = get_n_max(self.config.environment.components[etype])
            controller_kwargs = extend_kwargs(config.controller_kwargs, n_max)
            kwargs[etype] = controller_kwargs
        return create_dataclass_from_dict('ControllerParameters', kwargs)

    def create_controllers(self):
        controllers = {}
        for etype, config in self.config.client.items():
            component_config = self.config.environment.components[etype]
            n_max = get_n_max(component_config)
            cls = hydra.utils.get_class(config.cls)
            kwargs = extend_kwargs(config.controller_kwargs, n_max)
            controllers[etype] = cls(etype, **kwargs)
        return controllers
