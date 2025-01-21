import os 
import logging

import hydra

from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

abs_config_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../conf"))
config_dir_path = os.path.relpath(abs_config_dir_path, start=os.path.dirname(__file__))


def load_default_config() -> DictConfig:
    """Load the default scene configuration

    :return: default scene configuration
    """
    with initialize(config_path=config_dir_path, version_base=None):
        cfg = compose(config_name="config")
        scene_config = OmegaConf.merge(cfg.default, cfg.scene)
    return scene_config


def load_scene_config(scene_name: str) -> DictConfig:
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
        return scene_config
