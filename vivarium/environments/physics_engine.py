from functools import partial

import jax
import jax.numpy as jnp

from jax import vmap, lax
from jax_md import rigid_body, util, simulate, energy, quantity

f32 = util.f32

SPACE_NDIMS = 2


def friction_state_fn(mask_fn):
    def state_fn(state, neighbor):
        mask = mask_fn(state)
        # Issue : We need to sum the forces, here we just set them (so previous force functions in the env actually not used)
        return state.set(entities=state.entities.set(force=friction_force(state, neighbor, mask)))
    return state_fn


def to_rigid_body(position):
    return rigid_body.RigidBody(center=position, orientation=jnp.zeros(position.shape[0]))


def handle_rigid_body(force_fn):
    def wrapped_force_fn(state, neighbor, exists_mask):
        force = force_fn(state, neighbor, exists_mask)
        return to_rigid_body(force) if state.entities.is_rigid_body() and not isinstance(force,rigid_body.RigidBody) else force
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
    coll_force_fn = quantity.force(
        partial(total_collision_energy, displacement=displacement)
    )
    
    @handle_rigid_body
    def force_fn(state, neighbor, exists_mask):
        """Returns the collision force function of the environment

        :param state: state
        :param neighbor: neighbor maps of entities
        :param exists_mask: mask on existing entities
        :return: collision force function
        """
        return coll_force_fn(
            state.entities.unified_position,
            neighbor=neighbor,
            exists_mask=exists_mask,
            diameter=state.entities.diameter,
            epsilon=state.collision_eps,
            alpha=state.collision_alpha,
        )
        
    return force_fn


def collision_state_fn(displacement, mask_fn):
    coll_fn = collision_force_fn(displacement)
    def state_fn(state, neighbor):
        mask = mask_fn(state)
        force = coll_fn(state, neighbor, mask)
        if state.entities.is_rigid_body():
            force = force.set(center=state.entities.force.center + force.center,
                              orientation=state.entities.force.orientation + force.orientation)
        else:
            force = state.entities.force + force
        entities=state.entities.set(force=force)
        return state.set(entities=entities)
    return state_fn


# Functions to compute the verlet force on the whole system
@handle_rigid_body
def friction_force(state, neighbor, exists_mask):
    """Compute the friction force on the system

    :param state: current state of the system
    :param exists_mask: mask to specify which particles exist
    :return: friction force on the system
    """
    cur_vel = state.entities.unified_momentum / state.entities.unified_mass
    # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities)
    mask = jnp.stack([exists_mask] * 2, axis=1)
    cur_vel = jnp.where(mask, cur_vel, 0.0)
    return -jnp.tile(state.entities.friction, (SPACE_NDIMS, 1)).T * cur_vel
    

def friction_force_fn(displacement):
    return friction_force


def friction_state_fn(mask_fn):
    def state_fn(state, neighbor):
        mask = mask_fn(state)
        force = friction_force(state, neighbor, mask)
        if state.entities.is_rigid_body():
            force = force.set(center=state.entities.force.center + force.center,
                              orientation=state.entities.force.orientation + force.orientation)
        else:
            force = state.entities.force + force
        entities=state.entities.set(force=force)
        return state.set(entities=entities)
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


def verlet_force_fn(displacement):
    """Compute the verlet force on the whole system

    :param displacement: displacement function of jax_md
    :return: force function of the system
    """

    return sum_force_fns(displacement, [collision_force_fn, friction_force_fn])


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


def reset_force_state_fn():
    def fn(state, neighbor):
        if state.entities.is_rigid_body():
            zeros = to_rigid_body(jnp.zeros_like(state.entities.force.center))
        else:
            zeros = jnp.zeros_like(state.entities.force)
        return state.set(entities=state.entities.set(force=zeros))
    return fn


def init_state_fn(key, kT=0.0):
    key_cpy = key
    def fn(state):
        assert state.entities.momentum is None
        key, new_key = jax.random.split(key_cpy)
        assert not jnp.any(state.entities.unified_force) 
        if state.entities.is_rigid_body():
            assert not jnp.any(state.entities.force.orientation)
        return state.set(entities=simulate.initialize_momenta(state.entities, new_key, kT))
        
    return fn


def step_state_fn(shift, mask_fn, key, kT=0.0):
    
    def state_fn(state, neighbor):
        mask = mask_fn(state)

        dt_2 = state.dt / 2.0

        # Compute changes on entities
        new_force = state.entities.force
        entities=state.entities.set(force=state.entities.previous_force)
        entities = simulate.momentum_step(entities, dt_2)
        # TODO : why do we used dt and not dt/2 in the line below ?
        entities = simulate.position_step(
            entities, shift, dt_2, neighbor=neighbor
        )
        entities = entities.set(force=new_force)
        entities = entities.set(previous_force=new_force)
        entities = simulate.momentum_step(entities, dt_2)
        entities = mask_momentum(entities, mask)
        return state.set(entities=entities)
    return state_fn
