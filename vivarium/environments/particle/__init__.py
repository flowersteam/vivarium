import jax.numpy as jnp
import numpy as np

from jax import random

from vivarium.environments.particle.state import (
    State,
    ParticleState
)

from vivarium.environments.braitenberg.point_particle.init import init_entities

# Constants
SEED = 0
MAX_PARTICLES = 10
N_DIMS = 2
BOX_SIZE = 100
MASS = 1.0
DIAMETER = 0.2
NEIGHBOR_RADIUS = 100.0
COLLISION_ALPHA = 2.
COLLISION_EPS = 1.
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

def init_particles(
    entity_idx_offset=0,
    max_particles=MAX_PARTICLES,
    key=random.PRNGKey(SEED),
    **kwargs):
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
        ent_idx=jnp.array(range(entity_idx_offset, entity_idx_offset + max_particles)), 
        # entity_type=jnp.full((max_particles,), entity_type),
        idx=jnp.array(range(max_particles)),
        **kwargs
    )


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
    key, s_key = random.split(key, 2)

    ent_sub_types = {'AGENTS': (0, 0), 'PARTICLES': (1, max_particles)}

    entities = init_entities(
        max_agents=0,
        max_objects=max_particles,
        ent_sub_types=ent_sub_types,  # e.g. {'PREYS': (0, 5), 'PREDS': (1, 5), 'RESOURCES': (2, 5), 'POISON': (3, 5)}
        n_dims=N_DIMS,
        box_size=box_size,
        existing_agents=None,
        existing_objects=None,
        mass_center=mass,
        mass_orientation=1.,
        diameter=diameter,
        friction=friction,
        agents_pos=None,
        objects_pos=None,
        key=s_key)

    particles = init_particles(
        entity_idx_offset=0,
        max_particles=max_particles,
        **kwargs
    )

    return State(time=0, box_size=box_size, max_particles=max_particles, 
                 neighbor_radius=neighbor_radius, dt=dt, 
                 collision_alpha=collision_alpha, collision_eps=collision_eps,
                 entities=entities, objects=particles)


