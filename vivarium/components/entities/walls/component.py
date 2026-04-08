import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md.energy import soft_sphere
from jax_md import quantity

from .geometry import batch_segment_point_distance
from vivarium.components.component import Component

# Note: unsure if walls spatial attributes should be part of the EntityState or not
# (if so they will be part of nearest neighbor computation)
# Currently they are not


@md_dataclass
class WallState:
    epsilon: jnp.ndarray
    alpha: jnp.ndarray
    coordinates: jnp.ndarray
    

class WallComponent(Component):
    def __init__(self, name, precedence, epsilon, alpha, wall_coordinates):
        super().__init__(name, precedence)
        self.entity_type = name
        self.wall_coordinates = jnp.array(wall_coordinates)
        self.n_max = len(wall_coordinates)
        self.epsilon = epsilon
        self.alpha = alpha

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'epsilon': getattr(state, self.entity_type).epsilon.item(), #if isinstance(getattr(state, self.entity_type).epsilon, jnp.ndarray) else getattr(state, self.entity_type).epsilon,
            'alpha': getattr(state, self.entity_type).alpha.item(),
            'wall_coordinates': self.wall_state.wall_coordinates.tolist(),
        })
        return config


    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.entity_type] = WallState
        setattr(state_cls, self.entity_type, None)
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        # entity_idx = jnp.arange(0, self.n_max, dtype=int)
        wall_state = state.__annotations__[self.entity_type](
                        #    entity_idx=entity_idx,
                           coordinates=self.wall_coordinates,
                           epsilon=jnp.array(self.epsilon),
                           alpha=jnp.array(self.alpha)
        )

        return state.set(
            **{self.entity_type: wall_state}
        )

    def get_step_function(self, state, neighbor_manager, key):
        
        def state_fn(state, neighbor, key):
        
            def collision_energy(entity_positions):
                has_ortho, distances, normals = batch_segment_point_distance(
                    getattr(state, self.entity_type).coordinates, 
                    entity_positions
                    )
                energy = soft_sphere(
                    dr=distances,
                    sigma=state.entity_state.diameter[:, jnp.newaxis] / 2.,
                    epsilon=getattr(state, self.entity_type).epsilon,
                    alpha=getattr(state, self.entity_type).alpha
                )
                energy = energy * has_ortho
                return energy.sum()

            force = quantity.force(collision_energy)(state.entity_state.position)
            
            return state.set(
                entity_state=state.entity_state.set(
                    force=force + state.entity_state.force
                    )
            )
        return state_fn
