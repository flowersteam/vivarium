import os 
import random
import logging
from math import pi
from dataclasses import dataclass, fields
from collections.abc import Iterable

from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import hydra
from typing import Type

import jax.numpy as jnp

from vivarium.environments.state import create_state_cls, to_rigid_body_state
from vivarium.controllers.dataclass_wrapper import create_dataclass_from_dict
from vivarium.utils.converters import import_class


# from vivarium.controllers.simulator_controller import ControllerEntity
# from vivarium.controllers.panel_controller import ParamEntity
# from vivarium.controllers.simulator_controller import ControllerEntity
# from vivarium.controllers.panel_controller import  PanelControllerEntity


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

        return scene_config

def extend_kwargs(kwargs, n_exists):
    """Extend kwargs to n_exists items"""
    for attr, val in kwargs.items():
        if isinstance(val, Iterable) and '_all_values_' in val:
            kwargs[attr] = [val['_all_values_']] * n_exists
    return kwargs


@dataclass
class ConstructorConfiguration:
    cls: Type
    kwargs: dict


@dataclass
class SimulatorConfiguration:
    freq: float
    box_size: float
    num_scan_steps: int
    neighbor_radius: float
    to_jit: bool

    @classmethod
    def from_simulator(cls, simulator):
        field_names = [field.name for field in fields(cls)]
        return cls(**{
            field_name: getattr(simulator, field_name)
            for field_name in field_names})


@dataclass
class EntityTypeConfiguration:
    name: str
    idx: int
    kwargs: dict
    state_cls: Type


@dataclass
class EntityTypeClientConfiguration:
    param: ConstructorConfiguration
    simulator_controller: ConstructorConfiguration
    notebook_controller: ConstructorConfiguration
    panel_controller: ConstructorConfiguration
    panel_interface: ConstructorConfiguration


class SceneConfiguration:

    def __init__(self, scene_name: str, seed=None):
        self.scene_name = scene_name
        self.config = load_scene_config(scene_name)
        self._base_state_cls = import_class(self.config.state.cls)
        self.entity_state_cls = import_class(self.config.entities.entity_state.cls)
        self.entity_type_configs = {
            entity_type: EntityTypeConfiguration(
                name=entity_type,
                idx=idx,
                kwargs=extend_kwargs(self.config.entities[entity_type].kwargs,
                                     self.config.entities[entity_type].kwargs.n_exists),
                state_cls=import_class(self.config.entities[entity_type].cls)
            )
            for idx, entity_type in enumerate(self.config.entities.entity_types)}
        if self.config.client != 'None':
            controller_parameters = self.create_controller_parameters()
            self.entity_type_client_configs = {
                entity_type: EntityTypeClientConfiguration(**{attr: ConstructorConfiguration(
                    cls= import_class(data.cls),
                    kwargs={'controller_parameters': getattr(controller_parameters, entity_type)}
                    ) for attr, data in config.items() if attr != 'kwargs'})
                for entity_type, config in self.config.client.items()}

        self.subtype_labels = {i: label for i, label in enumerate(self.config.subtypes)}
        self.entity_types = self.config.entities.entity_types
        self.seed = seed
        self.generate_missing_params()
        self.state = self.create_state()

    def generate_missing_params(self):
        for entity_type in self.config.entities.entity_types:
            entity_params = self.config.entities[entity_type].kwargs
            n = entity_params.n_exists
            if entity_params.position == 'random':
                # Generate random positions if not provided
                entity_params.position = generate_random_positions(n, self.config.environment.kwargs.box_size, self.seed)
            if entity_params.orientation == 'random':
                # Generate random orientations if not provided
                entity_params.orientation = generate_random_orientations(n, self.seed)
            self.config.entities[entity_type].kwargs = entity_params

    def create_state_cls(self):
        """Create the state from the scene configuration

        :return: The created state
        """

        State = create_state_cls(
            base_state_cls=self._base_state_cls,
            entity_types=self.entity_types,
            entity_state_cls=self.entity_state_cls,
            entity_types_to_cls={etype: config.state_cls for etype, config in self.entity_type_configs.items()}
        )

        return State
    
    def create_state(self, rigid_body=False):

        entity_idx_offset = 0
        etype_instance = {}
        entity_types_kwargs = {etype: config.kwargs for etype, config in self.entity_type_configs.items()}
        for etype in self.entity_types:
            cls = self.entity_type_configs[etype].state_cls
            etype_instance[etype] = cls.create(entity_idx_offset, entity_types_kwargs, etype)
            entity_idx_offset += self.entity_type_configs[etype].kwargs['n_exists']

        state_cls = self.create_state_cls()
        state = state_cls(**{attr: jnp.array(val) for attr, val in self.config.state.kwargs.items()},
                        entity_state = self.entity_state_cls.create(self.entity_types, entity_types_kwargs),  #, etype_order), 
                        **etype_instance)
        if rigid_body:
            state = state.set(entity_state = to_rigid_body_state(state.entity_state))
        return state
    
    def create_environment(self, state=None):
        state = state or self.state
        env_cls = import_class(self.config.environment.cls)
        return env_cls(state=state, **self.config.environment.kwargs)
    
    def create_simulator(self, state=None, env=None):
        state = state or self.state
        # What about the case where state is not None and env is None?
        env = env or self.create_environment(state=state)
        simulator_cls = import_class(self.config.simulator.cls)
        return simulator_cls(env=env, scene_name=self.scene_name, 
                             controller_parameters=self.create_controller_parameters(), 
                             **self.config.simulator.kwargs)

    def create_controller_parameters(self):
        if self.config.client == 'None':
            return None
        kwargs = {entity_type: extend_kwargs(config.kwargs, self.config.entities[entity_type].kwargs.n_exists) for entity_type, config in self.config.client.items()}
        return create_dataclass_from_dict('ControllerParameters', kwargs)