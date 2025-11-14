import jax.numpy as jnp

from vivarium.environment.components.entities.controller import EntityController
from vivarium.environment.components.component import Component
from vivarium.utils.scene_configs import compute_parameters


#TODO: exists masks in entity components are currently handled manually and quite on a case-by-case basis.
# However, jax_md.partition.neighbor_list has a custom_mask_function argument. 
# Should we use it to simplify the code and make it less error prone?


class EntityComponent(Component):

    controller_cls = EntityController

    def __init__(self, name, precedence, entity_type, subtype,
                 position, orientation, mass, diameter, friction, exists, subtype_labels=None
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
        self.subtype_labels = subtype_labels

        self.is_entity_component = True  # No longer needed?

    @classmethod
    def from_config(cls, config, **kwargs):

        kwargs.update(cls.get_kwargs(config))

        if 'entity_type' not in kwargs:
            kwargs['entity_type'] = kwargs['name']

        return cls(subtype_labels=config.subtype_labels, **kwargs)

    def to_config(self, state):
        config = super().to_config(state)

        config.update({
            'n_max': state.exists(self.entity_type).shape[0],
            'exists': [bool(e.item()) for e in state.exists(self.entity_type)],
            'mass': state.mass(self.entity_type)[:, 0].tolist(),
            'position': state.position(self.entity_type).tolist(),
            'orientation': state.orientation(self.entity_type).tolist(),
            'diameter': state.diameter(self.entity_type).tolist(),
            'friction': state.friction(self.entity_type).tolist(),
            'subtype': [self.subtype_labels[i.item()] for i in state.entity_subtype(self.entity_type)],
            'subtype_labels': self.subtype_labels
        })
        return config

    @staticmethod
    def get_kwargs(config, exclude=[]):
        kwargs = super(EntityComponent, EntityComponent).get_kwargs(config)
                            
        if 'n_exists' in config:
            n_exists = config.n_exists
            kwargs['exists'] = [i < n_exists for i in range(config['n_max'])]
        
        config.update(kwargs)
        kwargs.update(compute_parameters(config))

        exclude = exclude + ['_target_', 'n_max', 'n_exists', 'subtype_labels', 'by_indices', 'client']
        kwargs = {k: v for k, v in config.items()
                    if k not in exclude}

        kwargs['subtype'] = jnp.array([config.subtype_labels.index(s) for s in kwargs['subtype']])

        return kwargs

    def init_base_entity(self, entity_state):
        self.entity_type_int = jnp.array(entity_state.entity_type.max() + 1 if len(entity_state.entity_type) > 0 else 0)
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
