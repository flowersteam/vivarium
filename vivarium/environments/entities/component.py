import jax.numpy as jnp

from omegaconf import OmegaConf

from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environments.physics_engine import Component

from vivarium.utils.scene_configs import generate_random_positions, generate_random_orientations, get_n_max, extend_kwargs


def compute_parameters(config):
    assert 'n_exists' in config or 'n_max' in config, f"Either n_max ot n_exists has to be defined"
    n_max = get_n_max(config)
    n_exists = config.n_exists if 'n_exists' in config else n_max
    OmegaConf.update(config, 'n_max', n_max, force_add=True)
    OmegaConf.update(config, 'n_exists', n_exists, force_add=True)
    n = config.n_max
    if '_range_' in config.position:
        # Generate random positions within a specified range
        config.position = generate_random_positions(n, config.position['_range_'])  # , self.seed)
    if config.orientation == '_random_':
        # Generate random orientations if not provided
        config.orientation = generate_random_orientations(n)  # , self.seed)

    config = extend_kwargs(config, n)

    return config




class EntityComponent(Component):

    controller_cls = ControllerEntity

    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction, exists
                 ):
        super().__init__(name, precedence)
        self.n_max = len(position)
        self.entity_type = entity_type
        self.subtype = subtype
        self.position = jnp.array(position)
        self.orientation = jnp.array(orientation)
        self.mass = jnp.array(mass)
        self.diameter = jnp.array(diameter)
        self.friction = jnp.array(friction)
        self.exists = jnp.array(exists)

        self.is_entity_component = True

    @classmethod
    def from_config(cls, config, **kwargs):

        kwargs.update(cls.get_kwargs(config))

        if 'entity_type' not in kwargs:
            kwargs['entity_type'] = kwargs['name']

        return cls(**kwargs)

    def to_config(self, state):
        config = super().to_config(state)

        n_subtypes = max(state.entity_state.entity_subtype) + 1
        subtype_to_n = [[i, sum(state.entity_subtype(self.entity_type) == i).item()] for i in range(n_subtypes)]

        config.update({
            'n_exists': sum(state.exists(self.entity_type)).item(),  #TODO: associations between entity existence, subtypes and other attributes might get mixed up, to fix
            'mass': state.mass(self.entity_type)[:, 0].tolist(),
            'position': state.position(self.entity_type).tolist(),
            'orientation': state.orientation(self.entity_type).tolist(),
            'diameter': state.diameter(self.entity_type).tolist(),
            'friction': state.friction(self.entity_type).tolist(),
            'subtype_to_n': subtype_to_n
        })
        return config

    @staticmethod
    def get_kwargs(config, exclude=[]):
        kwargs = super(EntityComponent, EntityComponent).get_kwargs(config)
        config.update(kwargs)
        kwargs.update(compute_parameters(config))

        n_max = len(config.position)
        exclude = exclude + ['_target_', 'n_max', 'n_exists', 'subtype_to_n']
        kwargs = {k: v for k, v in config.items()
                    if k not in exclude}
        exists = jnp.zeros(n_max, dtype=int)
        exists = exists.at[:config.n_exists].set(1)
        kwargs['exists'] = exists
        subtype = []
        for s, n in config.subtype_to_n:
            subtype.extend([s] * n)
        
        kwargs['subtype'] = jnp.array(subtype)    
        
        return kwargs

    def init_base_entity(self, entity_state):
        self.entity_type_int = entity_state.entity_type.max() + 1 if len(entity_state.entity_type) > 0 else 0
        self.offset = entity_state.exists.shape[0]
        return entity_state.add_new_entities(
            positions=self.position,
            orientations=self.orientation,
            mass=self.mass,
            diameter=self.diameter,
            friction=self.friction,
            exists=self.exists,
            entity_type=jnp.full((self.n_max,), self.entity_type_int),
            entity_subtype=self.subtype,
            entity_type_idx=jnp.arange(self.n_max),
        )
