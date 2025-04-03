import os 
import re
import random
import logging
import importlib
from math import pi

from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import hydra


abs_config_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../conf"))
config_dir_path = os.path.relpath(abs_config_dir_path, start=os.path.dirname(__file__))


def generate_random_positions(n, box_size, seed=None):
    # Generate random positions within the box size
    rng = random.Random(seed)
    return [[rng.uniform(0, box_size), rng.uniform(0, box_size)] for _ in range(n)]


def generate_random_orientations(n, seed=None):
    # Generate random orientations
    rng = random.Random(seed)
    return [rng.uniform(0, 2 * pi) for _ in range(n)]


def load_default_config() -> DictConfig:
    """Load the default scene configuration

    :return: default scene configuration
    """
    with initialize(config_path=config_dir_path, version_base=None):
        cfg = compose(config_name="config")
        scene_config = OmegaConf.merge(cfg.default, cfg.scene)
    return scene_config


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
        scene_config = OmegaConf.merge(cfg.default, cfg.scene)
        for entity_type in scene_config['state_data']['entity_state']['entity_types']:
            entity_params = scene_config['state_data'][entity_type]['kwargs']
            n = entity_params['n_exists']
            if entity_params['position'] == 'None':
                # Generate random positions if not provided
                entity_params['position'] = generate_random_positions(n, scene_config['environment']['kwargs']['box_size'], seed)
            if entity_params['orientation'] == 'None':
                # Generate random orientations if not provided
                entity_params['orientation'] = generate_random_orientations(n, seed)
            scene_config['state_data'][entity_type]['kwargs'] = entity_params
        return scene_config


def import_class(class_path):
    if not isinstance(class_path, str):
        return class_path
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def class_import_path(cls) -> str:
    """Get the import path of a given class

    :param cls: The class to get the import path for
    :return: The import path as a string
    """
    return f"{cls.__module__}.{cls.__name__}"


def upper_camel_to_snake(name: str) -> str:
    """Convert an upper camel case string to snake case

    :param name: The string in upper camel case
    :return: The string in snake case
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def snake_to_upper_camel(name: str) -> str:
    """Convert a snake case string to upper camel case

    :param name: The string in snake case
    :return: The string in upper camel case
    """
    return ''.join(word.capitalize() for word in name.split('_'))
