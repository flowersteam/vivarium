import jax.numpy as jnp

from jax_md import energy, partition, quantity, smap, space
from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.utils import neighbors_entity_mask
from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import f32


def collision_force_fn(displacement, state_attr):

    def coll_force_fn(sigma, epsilon, alpha):
        """Compute the collision force on the system

        :param positions: positions of all the particles
        :param sigma: diameters of all the particles
        :param epsilon: interaction energy scale between two particles
        :param alpha: interaction stiffness between two particles
        :return: collision force on the system
        """
        return quantity.force(
            smap.pair_neighbor_list(
                energy.soft_sphere,
                space.canonicalize_displacement_or_metric(displacement),
                sigma=sigma,
                epsilon=epsilon,
                alpha=alpha
            )
        )

    def force_fn(state, neighbor, exists_mask):
        """Returns the collision force function of the environment

        :param state: state
        :param neighbor: neighbor maps of entities
        :param exists_mask: mask on existing entities
        :return: collision force function
        """

        collision_state = getattr(state, state_attr)

        #TODO: filter sources and targets based on their existence
        fn = coll_force_fn(
            sigma=(state.entity_state.diameter[:, jnp.newaxis] + state.entity_state.diameter[neighbor.idx]) / 2.,
            epsilon=collision_state.epsilon,
            alpha=collision_state.alpha
        )


        force = fn(
            state.entity_state.position,
            neighbor.set(
                idx = jnp.where(
                    neighbors_entity_mask(
                        neighbor.idx,
                        state.entity_state.exists,
                        state.entity_state.exists,
                        partition.neighbor_list_mask(neighbor, mask_self=True)
                    ),
                    neighbor.idx,
                    state.entity_state.exists.shape[0]
                )
            )
        )

        force = jnp.where(exists_mask[:, jnp.newaxis], force, jnp.array([0., 0.]))

        return force

    return force_fn

@md_dataclass
class CollisionState:
    epsilon: jnp.ndarray
    alpha: jnp.ndarray

class CollisionComponent(Component):
    def __init__(self, name, precedence, epsilon, alpha, mask_fn):
        super().__init__(name, precedence)
        self.epsilon = epsilon
        self.alpha = alpha
        self.mask_fn = mask_fn
        self.state_attr = f'{self.name}_state'

    def to_config(self, state):
        config = super().to_config(state)
        collision_state = getattr(state, self.state_attr)
        config.update({
            'epsilon': collision_state.epsilon.item() if isinstance(collision_state.epsilon, jnp.ndarray) else collision_state.epsilon,
            'alpha': collision_state.alpha.item() if isinstance(collision_state.alpha, jnp.ndarray) else collision_state.alpha,
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            **{self.state_attr: CollisionState(
                epsilon=jnp.array(self.epsilon),
                alpha=jnp.array(self.alpha)
            )}
        )

    def update_state_cls(self, state_cls):
        state_cls.__annotations__[self.state_attr] = CollisionState
        setattr(state_cls, self.state_attr, None)
        return state_cls        

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        coll_fn = collision_force_fn(self.displacement, self.state_attr)
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)
            force = coll_fn(state, neighbor, mask)
            force = state.entity_state.force + force
            entity_state=state.entity_state.set(force=force)
            return state.set(entity_state=entity_state)
        return state_fn
