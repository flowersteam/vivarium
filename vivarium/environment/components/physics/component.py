import logging

from jax import vmap, lax
import jax.numpy as jnp
import jax

from jax_md import rigid_body, util, simulate, energy, quantity, smap, space, partition

from vivarium.environment.components.component import Component
from vivarium.environment.utils import neighbors_entity_mask, get_relative_displacement


lg = logging.getLogger(__name__)

f32 = util.f32

SPACE_NDIMS = 2

#TODO: exists masks are currently handled manually and quite on a case-by-case basis.
# However, jax_md.partition.neighbor_list has a custom_mask_function argument. 
# Should we use it to simplify the code and make it less error prone?


def to_rigid_body(position):
    return rigid_body.RigidBody(center=position, orientation=jnp.zeros(position.shape[0]))


def handle_rigid_body(force_fn):
    def wrapped_force_fn(state, neighbor, exists_mask):
        force = force_fn(state, neighbor, exists_mask)
        return to_rigid_body(force) if state.entity_state.is_rigid_body() and not isinstance(force,rigid_body.RigidBody) else force
    return wrapped_force_fn


def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha, mask):
    """Compute the collision energy between a pair of particles

    :param displacement_fn: displacement function of jax_md
    :param r_a: position of particle a
    :param r_b: position of particle b
    :param l_a: diameter of particle a
    :param l_b: diameter of particle b
    :param epsilon: interaction energy scale
    :param alpha: interaction stiffness
    :param mask: set the energy to 0 if one of the particles is masked
    :return: collision energy between both particles
    """
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = (l_a + l_b) / 2
    e = energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=f32(alpha))
    return jnp.where(mask, e, 0.0)


collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None, 0))


def total_collision_energy(
    positions, diameter, neighbor, displacement, exists_mask, epsilon, alpha
):
    """Compute the collision energy between all neighboring pairs of particles in the system

    :param positions: positions of all the particles
    :param diameter: diameters of all the particles
    :param neighbor: neighbor array of the system
    :param displacement: dipalcement function of jax_md
    :param exists_mask: mask to specify which particles exist
    :param epsilon: interaction energy scale between two particles
    :param alpha: interaction stiffness between two particles
    :return: sum of all collisions energies of the system
    """
    diameter = lax.stop_gradient(diameter)
    senders, receivers = neighbor.idx

    r_senders = positions[senders]
    r_receivers = positions[receivers]
    l_senders = diameter[senders]
    l_receivers = diameter[receivers]

    # Set collision energy to zero if the sender or receiver is non existing
    mask = exists_mask[senders] * exists_mask[receivers]

    energies = collision_energy(
        displacement,
        r_senders,
        r_receivers,
        l_senders,
        l_receivers,
        epsilon,
        alpha,
        mask,
    )

    return jnp.sum(energies)


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


@handle_rigid_body
def friction_force(state, neighbor, exists_mask):
    """Compute the friction force on the system

    :param state: current state of the system
    :param exists_mask: mask to specify which particles exist
    :return: friction force on the system
    """
    cur_vel = state.entity_state.unified_momentum / state.entity_state.unified_mass
    # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities)
    mask = jnp.stack([exists_mask] * 2, axis=1)
    cur_vel = jnp.where(mask, cur_vel, 0.0)
    return -jnp.tile(state.entity_state.friction, (SPACE_NDIMS, 1)).T * cur_vel
    

class FrictionComponent(Component):
    def __init__(self, name, precedence, mask_fn):
        super().__init__(name, precedence)
        self.mask_fn = mask_fn

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def get_step_function(self, state, neighbor_manager, key):
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)
            force = friction_force(state, neighbor, mask)
            if state.entity_state.is_rigid_body():
                force = force.set(center=state.entity_state.force.center + force.center,
                                orientation=state.entity_state.force.orientation + force.orientation)
            else:
                force = state.entity_state.force + force
            entity_state=state.entity_state.set(force=force)
            return state.set(entity_state=entity_state)
        return state_fn


def sum_forces(force_list):
    if isinstance(force_list[0], rigid_body.RigidBody):
        return rigid_body.RigidBody(center=jnp.array([f.center for f in force_list]).sum(0), 
                                    orientation=jnp.array([f.orientation for f in force_list]).sum(0))
    return jnp.array(force_list).sum(0)


def sum_force_fns(displacement, force_fns):
    fns = [fn(displacement) for fn in force_fns]
    def force_fn(state, neighbor, exists_mask):
        force = sum_forces([fn(state, neighbor, exists_mask) for fn in fns])
        return force
    return force_fn


class ResetForceComponent(Component):

    def get_step_function(self, state, neighbor_manager, key):
        def fn(state, neighbor, key):
            if state.entity_state.is_rigid_body():
                zeros = to_rigid_body(jnp.zeros_like(state.entity_state.force.center))
            else:
                zeros = jnp.zeros_like(state.entity_state.force)
            return state.set(entity_state=state.entity_state.set(force=zeros))
        return fn


def mask_momentum(entity_state, exists_mask):
    """
    Set the momentum values to zeros for non existing entities
    :param entity_state: entity_state
    :param exists_mask: bool array specifying which entities exist or not
    :return: entity_state: new entities state state with masked momentum values
    """
    
    exists_mask_space = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1)
    momentum = jnp.where(exists_mask_space, entity_state.unified_momentum, 0)
    if entity_state.is_rigid_body():
        orientation = jnp.where(exists_mask, entity_state.momentum.orientation, 0)
        momentum = rigid_body.RigidBody(center=momentum, orientation=orientation)
    return entity_state.set(momentum=momentum)


def init_state_fn(key, kT=0.0):
    key_cpy = key
    def fn(state):
        assert state.entity_state.momentum is None
        key, new_key = jax.random.split(key_cpy)
        assert not jnp.any(state.entity_state.unified_force) 
        if state.entity_state.is_rigid_body():
            assert not jnp.any(state.entity_state.force.orientation)
        return state.set(entity_state=simulate.initialize_momenta(state.entity_state, new_key, kT))
        
    return fn


class StepComponent(Component):
    def __init__(self, name, precedence, dt, mask_fn):
        super().__init__(name, precedence)
        self.dt = dt
        self.mask_fn = mask_fn

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'dt': state.dt.item() if isinstance(state.dt, jnp.ndarray) else state.dt,
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def update_state_cls(self, state_cls):
        state_cls.__annotations__['dt'] = jnp.float32
        state_cls.dt = None
        return state_cls
    
    def init_state_fn(self, state, neighbor_manager, key):
        if state.entity_state.momentum is None:
            key, sub_key = jax.random.split(key)
            state = init_state_fn(sub_key)(state)
        return state.set(dt=self.dt)

    def get_step_function(self, state, neighbor_manager, key):
        self.shift = neighbor_manager.shift
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)

            dt_2 = state.dt / 2.0

            # Compute changes on entities
            new_force = state.entity_state.force
            entity_state=state.entity_state.set(force=state.entity_state.previous_force)
            entity_state = simulate.momentum_step(entity_state, dt_2)
            # TODO : why do we used dt and not dt/2 in the line below ?
            entity_state = simulate.position_step(
                entity_state, self.shift, dt_2, neighbor=neighbor
            )
            entity_state = entity_state.set(force=new_force)
            entity_state = entity_state.set(previous_force=new_force)
            entity_state = simulate.momentum_step(entity_state, dt_2)
            entity_state = mask_momentum(entity_state, mask)
            return state.set(entity_state=entity_state)
        return state_fn


class ProximityMapComponent(Component):

    def init_state_fn(self, state, neighbor_manager, key):
        fn = self.get_step_function(state, neighbor_manager, key)
        state = fn(state, neighbor_manager.neighbors, key)
        return state

    def update_state_cls(self, state_cls):
        state_cls.__annotations__['distance_map'] = jnp.ndarray
        state_cls.__annotations__['orientation_map'] = jnp.ndarray
        state_cls.distance_map = None
        state_cls.orientation_map = None
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        source_mask = jnp.full(state.entity_state.exists.shape, True, dtype=bool)
        def step_fn(state, neighbors, key):

            all_dist, all_relative_theta = (
                get_relative_displacement(
                    state.entity_state.position,
                    state.entity_state.orientation, 
                    source_mask,
                    neighbors.idx, 
                    displacement_fn=neighbor_manager.displacement
                )
            )

            return state.set(
                distance_map = all_dist,
                orientation_map = all_relative_theta
            )

        return step_fn
