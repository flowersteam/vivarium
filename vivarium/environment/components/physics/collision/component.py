import jax.numpy as jnp

from jax_md import energy, partition, quantity, smap, space

from vivarium.environment.utils import neighbors_entity_mask
from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import f32, handle_rigid_body


def collision_force_fn(displacement):

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

    @handle_rigid_body
    def force_fn(state, neighbor, exists_mask):
        """Returns the collision force function of the environment

        :param state: state
        :param neighbor: neighbor maps of entities
        :param exists_mask: mask on existing entities
        :return: collision force function
        """

        #TODO: filter sources and targets based on their existence
        fn = coll_force_fn(
            sigma=(state.entity_state.diameter[:, jnp.newaxis] + state.entity_state.diameter[neighbor.idx]), # / 2,
            epsilon=state.collision_eps,
            alpha=state.collision_alpha
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


class CollisionComponent(Component):
    def __init__(self, name, precedence, epsilon, alpha, mask_fn):
        super().__init__(name, precedence)
        self.epsilon = epsilon
        self.alpha = alpha
        self.mask_fn = mask_fn

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'epsilon': state.collision_eps.item() if isinstance(state.collision_eps, jnp.ndarray) else state.collision_eps,
            'alpha': state.collision_alpha.item() if isinstance(state.collision_alpha, jnp.ndarray) else state.collision_alpha,
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            collision_eps=self.epsilon,
            collision_alpha=self.alpha
            )

    def update_state_cls(self, state_cls):
        state_cls.__annotations__['collision_eps'] = f32
        state_cls.__annotations__['collision_alpha'] = f32
        state_cls.collision_eps = None
        state_cls.collision_alpha = None
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        coll_fn = collision_force_fn(self.displacement)
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)
            force = coll_fn(state, neighbor, mask)
            if state.entity_state.is_rigid_body():
                force = force.set(center=state.entity_state.force.center + force.center,
                                orientation=state.entity_state.force.orientation + force.orientation)
            else:
                force = state.entity_state.force + force
            entity_state=state.entity_state.set(force=force)
            return state.set(entity_state=entity_state)
        return state_fn
