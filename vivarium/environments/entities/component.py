import jax.numpy as jnp
from vivarium.controllers.simulator_controller import ControllerEntity
from vivarium.environments.physics_engine import Component


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
    def get_kwargs(cls, name, scene_config, config_node, exclude=[]):
        kwargs = scene_config.compute_parameters(name, config_node)
        n_max = len(config_node.position)
        exclude = exclude + ['_target_', 'n_max', 'n_exists', 'subtype_to_n']
        kwargs = {k: v for k, v in config_node.items()
                  if k not in exclude}
        exists = jnp.zeros(n_max, dtype=int)
        exists = exists.at[:config_node.n_exists].set(1)
        kwargs['exists'] = exists
        subtype = []
        for s, n in config_node.subtype_to_n:
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
