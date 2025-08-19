import os 
import random
import logging
from math import pi
from collections.abc import Iterable

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import hydra


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


def load_config(rel_config_dir_path, config_name, overrides=[]):
    path = os.path.join(config_dir_path, rel_config_dir_path)

    with hydra.initialize(config_path=path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        return cfg


def load_scene_config(scene_name: str) -> DictConfig:
    """Load a specific scene configuration

    :param scene_name: scene name of yaml file
    :return: scene configuration
    """
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    return load_config('scene', scene_name)


def extend_kwargs(kwargs, n):
    """Extend kwargs to n items"""
    for attr, val in kwargs.items():
        if isinstance(val, Iterable) and '_all_values_' in val:
            kwargs[attr] = [val['_all_values_']] * n
    return kwargs


def compute_parameters(config):
    n = config.n_max
    if '_range_' in config.position:
        # Generate random positions within a specified range
        config.position = generate_random_positions(n, config.position['_range_'])  # , self.seed)
    if config.orientation == '_random_':
        # Generate random orientations if not provided
        config.orientation = generate_random_orientations(n)  # , self.seed)

    config = extend_kwargs(config, n)

    return config


def component_factories_from_config(config):
    """Create component factories from a configuration object."""
    component_factories = [
        hydra.utils.get_class(c._target_).from_config(c,name=name) for name, c in config.items()
    ]
    return component_factories
