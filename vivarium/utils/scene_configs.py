import os
import random
from math import pi
from collections.abc import Iterable
from typing import Dict, List

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import hydra

from vivarium.utils.runtime import get_config_dir

OmegaConf.register_new_resolver("range", lambda start, end: list(range(start, end)))

abs_config_dir_path = get_config_dir()
config_dir_path = os.path.relpath(abs_config_dir_path, start=os.path.dirname(__file__))


def get_available_scenes() -> Dict[str, List[str]]:
    """List all available scene configurations, grouped by category.

    Returns:
        Dict with keys like 'Sessions', 'Tutorials', 'Research', 'Sandbox'
        and values as lists of scene names.
    """
    scene_dir = os.path.join(get_config_dir(), 'scene')
    exclude_patterns = ['base_scene', '_defaults', 'default', 'session_defaults', 'braitenberg_defaults']

    # Categorize scenes
    sessions = []
    tutorials = []
    research = []
    sandbox = []

    for filename in os.listdir(scene_dir):
        if filename.endswith('.yaml'):
            name = filename[:-5]
            if any(p in name for p in exclude_patterns):
                continue
            if name.startswith('session'):
                sessions.append(name)
            elif name in ['quickstart']:
                tutorials.append(name)
            elif name in ['sandbox', 'simple', 'custom_positions']:
                sandbox.append(name)
            else:
                research.append(name)

    return {
        'Sessions': sorted(sessions),
        'Tutorials': sorted(tutorials),
        'Research': sorted(research),
        'Sandbox': sorted(sandbox)
    }


def get_available_scenes_flat() -> List[str]:
    """List all available scene configurations as a flat list.

    Returns:
        List of scene names.
    """
    grouped = get_available_scenes()
    scenes = []
    for category_scenes in grouped.values():
        scenes.extend(category_scenes)
    return sorted(scenes)


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
        
        # Allow dynamically adding new fields to DictConfig
        # OmegaConf.set_struct(cfg, False)
        
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
            
        if attr == 'by_indices':
            for each in val:
                for label, data in each.items():
                    for k, v in data.items():
                        if k != 'indices' and k != 'client' and k != 'n_exists':
                            for idx in data.indices:
                                kwargs[k][idx] = v
                        elif k == 'n_exists':
                            for i, idx in enumerate(data.indices):
                                kwargs['exists'][idx] = i < data.n_exists
            
    return kwargs


def extend_controller_kwargs(kwargs, by_indices, n):
    kwargs = extend_kwargs(kwargs, n)
    for each in by_indices:
        for label, data in each.items():
            if 'client' in data:
                for k, v in data.client.items():
                    for idx in data.indices:
                        kwargs[k][idx] = v

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
        hydra.utils.get_class(c._target_).from_config(c, name=name) for name, c in config.component_list.items()
    ]
    return component_factories
