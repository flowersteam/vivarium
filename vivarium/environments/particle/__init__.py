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
NEIGHBOR_RADIUS = 100.0
#COLLISION_ALPHA = 0.5
#COLLISION_EPS = 0.1
DT = 0.1
MU_K = 4.0
SIGMA_K = 1.0
W_K = 0.022
MU_G = 0.6
SIGMA_G = 0.15
C_REP = 1.0


def init_state(
    box_size=BOX_SIZE,
    dt=DT,
    max_particles=MAX_PARTICLES,
    neighbor_radius=NEIGHBOR_RADIUS,
    # collision_alpha=COLLISION_ALPHA,
    # collision_eps=COLLISION_EPS,
    n_dims=N_DIMS,
    seed=SEED,
    mass=MASS,
    existing_particles=None,
    mu_k=MU_K,
    sigma_k=SIGMA_K,
    w_k=W_K,
    mu_g=MU_G,
    sigma_g=SIGMA_G,
    c_rep=C_REP
) -> State:

    key = random.PRNGKey(seed)
    key, key_pos = random.split(key, 2)

    particle_state = init_particle_state(
        max_particles=max_particles,
        n_dims=n_dims,
        box_size=box_size,
        existing_particles=existing_particles,
        mass=mass,
        key_pos=key_pos,
    )

    return State(time=0, box_size=box_size, max_particles=max_particles, neighbor_radius=neighbor_radius, dt=dt, particle_state=particle_state)


def init_particle_state(
    max_particles=MAX_PARTICLES,
    n_dims=N_DIMS,
    box_size=BOX_SIZE,
    existing_particles=None,
    mass=MASS,
    mu_k=MU_K,
    sigma_k=SIGMA_K,
    w_k=W_K,
    mu_g=MU_G,
    sigma_g=SIGMA_G,
    c_rep=C_REP,
    key_pos=random.PRNGKey(SEED)):

    existing_particles = existing_particles or max_particles
    
    particle_positions = random.uniform(key_pos, (max_particles, n_dims)) * box_size
    
    # Define arrays with existing entities
    exists_particles = jnp.concatenate(
        (jnp.ones((existing_particles)), jnp.zeros((max_particles - existing_particles)))
    )

    return ParticleState(
        position=particle_positions,
        momentum=None,
        force=jnp.zeros((max_particles, 2)),
        mass=jnp.full((max_particles, 1), mass),
        idx=jnp.array(range(max_particles)),
        exists=exists_particles,
        mu_k=jnp.full((max_particles,), MU_K),
        sigma_k=jnp.full((max_particles,), SIGMA_K),
        w_k=jnp.full((max_particles,), W_K),
        mu_g=jnp.full((max_particles,), MU_G),
        sigma_g=jnp.full((max_particles,), SIGMA_G),
        c_rep=jnp.full((max_particles,), C_REP)
    )
