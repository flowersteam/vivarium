import jax.numpy as jnp
import numpy as np

from jax import random

from vivarium.environments.particle.state import (
    State,
    ParticleState
)

# Constants
SEED = 0
MAX_PARTICLES = 10
N_DIMS = 2
BOX_SIZE = 100
MASS = 1.0
DIAMETER = 0.2
NEIGHBOR_RADIUS = 100.0
COLLISION_ALPHA = 0.5
COLLISION_EPS = 0.1
DT = 0.1
FRICTION = 1.

# Lenia parameters
LENIA_PARAMS = {
    'mu_k': 4.0,
    'sigma_k': 1.0,
    'w_k': 0.022,
    'mu_g': 0.6,
    'sigma_g': 0.15,
    'c_rep': 1.0
}


def init_state(
    box_size=BOX_SIZE,
    dt=DT,
    max_particles=MAX_PARTICLES,
    neighbor_radius=NEIGHBOR_RADIUS,
    collision_alpha=COLLISION_ALPHA,
    collision_eps=COLLISION_EPS,
    n_dims=N_DIMS,
    seed=SEED,
    diameter=DIAMETER,
    mass=MASS,
    friction=FRICTION,
    existing_particles=None,
    **kwargs
) -> State:

    key = random.PRNGKey(seed)
    #key, key_pos = random.split(key, 2)

    particle_state = init_particle_state(
        max_particles=max_particles,
        n_dims=n_dims,
        box_size=box_size,
        existing_particles=existing_particles,
        mass=mass,
        friction=friction,
        key=key,
        **kwargs
    )

    return State(time=0, box_size=box_size, max_particles=max_particles, 
                 neighbor_radius=neighbor_radius, dt=dt, 
                 collision_alpha=collision_alpha, collision_eps=collision_eps,
                 particle_state=particle_state)


def init_particle_state(
    max_particles=MAX_PARTICLES,
    n_dims=N_DIMS,
    box_size=BOX_SIZE,
    existing_particles=None,
    mass=MASS,
    friction=FRICTION,
    diameter=DIAMETER,
    # mu_k=MU_K,
    # sigma_k=SIGMA_K,
    # w_k=W_K,
    # mu_g=MU_G,
    # sigma_g=SIGMA_G,
    # c_rep=C_REP,
    key=random.PRNGKey(SEED),
    **kwargs):

    existing_particles = existing_particles or max_particles
     # Define arrays with existing entities
    exists_particles = jnp.concatenate(
        (jnp.ones((existing_particles)), jnp.zeros((max_particles - existing_particles)))
    )

    key, key_pos = random.split(key)   
    particle_positions = random.uniform(key_pos, (max_particles, n_dims)) * box_size
    

    for param, defaut_val in LENIA_PARAMS.items():
        val = kwargs[param] if param in kwargs else defaut_val
        if isinstance(val, float):
            kwargs[param] = jnp.full((max_particles,), val)
        elif isinstance(val, tuple) and len(val) == 2:
            mean, std = val
            key, key_param = random.split(key)
            kwargs[param] = random.normal(key_param, (max_particles,)) * std + mean
        elif isinstance(val, (np.ndarray, jnp.ndarray)):
            kwargs[param] = val


    return ParticleState(
        position=particle_positions,
        momentum=None,
        force=jnp.zeros((max_particles, 2)),
        mass=jnp.full((max_particles, 1), mass),
        diameter=jnp.full((max_particles,), diameter),
        friction=jnp.full((max_particles,), friction),
        idx=jnp.array(range(max_particles)),
        exists=exists_particles,
        **kwargs
    )
