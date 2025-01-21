import jax.numpy as jnp
import numpy as np

from jax import random
from jax_md.rigid_body import RigidBody

from vivarium.environments.braitenberg.behaviors import Behaviors, behavior_to_params
from vivarium.environments.braitenberg.simple.classes import (
    EntityType,
    State,
    AgentState,
    ObjectState,
    EntityState,
)

# Constants
SEED = 0
MAX_AGENTS = 10
MAX_OBJECTS = 2
N_DIMS = 2
BOX_SIZE = 100
DIAMETER = 5.0
FRICTION = 0.1
MASS_CENTER = 1.0
MASS_ORIENTATION = 0.125
NEIGHBOR_RADIUS = 100.0
COLLISION_ALPHA = 0.5
COLLISION_EPS = 0.1
DT = 0.1
WHEEL_DIAMETER = 2.0
SPEED_MUL = 1.0
MAX_SPEED = 10.0
THETA_MUL = 1.0
PROX_DIST_MAX = 40.0
PROX_COS_MIN = 0.0
AGENTS_COLOR = jnp.array([0.0, 0.0, 1.0])
OBJECTS_COLOR = jnp.array([1.0, 0.0, 0.0])
BEHAVIOR = Behaviors.AGGRESSION.value


def init_state(
    box_size=BOX_SIZE,
    dt=DT,
    max_agents=MAX_AGENTS,
    max_objects=MAX_OBJECTS,
    neighbor_radius=NEIGHBOR_RADIUS,
    collision_alpha=COLLISION_ALPHA,
    collision_eps=COLLISION_EPS,
    n_dims=N_DIMS,
    seed=SEED,
    diameter=DIAMETER,
    friction=FRICTION,
    mass_center=MASS_CENTER,
    mass_orientation=MASS_ORIENTATION,
    existing_agents=None,
    existing_objects=None,
    behavior=BEHAVIOR,
    wheel_diameter=WHEEL_DIAMETER,
    speed_mul=SPEED_MUL,
    max_speed=MAX_SPEED,
    theta_mul=THETA_MUL,
    prox_dist_max=PROX_DIST_MAX,
    prox_cos_min=PROX_COS_MIN,
    agents_color=AGENTS_COLOR,
    objects_color=OBJECTS_COLOR,
) -> State:

    key = random.PRNGKey(seed)
    key, key_agents_pos, key_objects_pos, key_orientations = random.split(key, 4)

    entities = init_entities(
        max_objects=max_objects,
        max_agents=max_agents,
        n_dims=n_dims,
        box_size=box_size,
        existing_agents=existing_agents,
        existing_objects=existing_objects,
        mass_center=mass_center,
        mass_orientation=mass_orientation,
        diameter=diameter,
        friction=friction,
        key_agents_pos=key_agents_pos,
        key_objects_pos=key_objects_pos,
        key_orientations=key_orientations,
    )

    agents = init_agents(
        max_agents=max_agents,
        behavior=behavior,
        wheel_diameter=wheel_diameter,
        speed_mul=speed_mul,
        max_speed=max_speed,
        theta_mul=theta_mul,
        prox_dist_max=prox_dist_max,
        prox_cos_min=prox_cos_min,
        agents_color=agents_color,
    )

    objects = init_objects(
        max_agents=max_agents, max_objects=max_objects, objects_color=objects_color
    )

    state = init_complete_state(
        entities=entities,
        agents=agents,
        objects=objects,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt,
    )

    return state


def init_entities(
    max_agents=MAX_AGENTS,
    max_objects=MAX_OBJECTS,
    n_dims=N_DIMS,
    box_size=BOX_SIZE,
    existing_agents=None,
    existing_objects=None,
    mass_center=MASS_CENTER,
    mass_orientation=MASS_ORIENTATION,
    diameter=DIAMETER,
    friction=FRICTION,
    key_agents_pos=random.PRNGKey(SEED),
    key_objects_pos=random.PRNGKey(SEED + 1),
    key_orientations=random.PRNGKey(SEED + 2),
):
    existing_agents = max_agents if not existing_agents else existing_agents
    existing_objects = max_objects if not existing_objects else existing_objects
    n_entities = (
        max_agents + max_objects
    )  # we store the entities data in jax arrays of length max_agents + max_objects
    # Assign random positions to each entity in the environment
    agents_positions = random.uniform(key_agents_pos, (max_agents, n_dims)) * box_size
    objects_positions = (
        random.uniform(key_objects_pos, (max_objects, n_dims)) * box_size
    )
    positions = jnp.concatenate((agents_positions, objects_positions))
    # Assign random orientations between 0 and 2*pi to each entity
    orientations = random.uniform(key_orientations, (n_entities,)) * 2 * jnp.pi
    # Assign types to the entities
    agents_entities = jnp.full(max_agents, EntityType.AGENT.value)
    object_entities = jnp.full(max_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)
    # Define arrays with existing entities
    exists_agents = jnp.concatenate(
        (jnp.ones((existing_agents)), jnp.zeros((max_agents - existing_agents)))
    )
    exists_objects = jnp.concatenate(
        (jnp.ones((existing_objects)), jnp.zeros((max_objects - existing_objects)))
    )
    exists = jnp.concatenate((exists_agents, exists_objects), dtype=int)

    return EntityState(
        position=RigidBody(center=positions, orientation=orientations),
        momentum=None,
        force=RigidBody(
            center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)
        ),
        mass=RigidBody(
            center=jnp.full((n_entities, 1), mass_center),
            orientation=jnp.full((n_entities), mass_orientation),
        ),
        entity_type=entity_types,
        entity_idx=jnp.array(list(range(max_agents)) + list(range(max_objects))),
        diameter=jnp.full((n_entities), diameter),
        friction=jnp.full((n_entities), friction),
        exists=exists,
    )


def init_agents(
    max_agents=MAX_AGENTS,
    behavior=BEHAVIOR,
    wheel_diameter=WHEEL_DIAMETER,
    speed_mul=SPEED_MUL,
    max_speed=MAX_SPEED,
    theta_mul=THETA_MUL,
    prox_dist_max=PROX_DIST_MAX,
    prox_cos_min=PROX_COS_MIN,
    agents_color=AGENTS_COLOR,
):
    # Need to use a np array because jax jax array can't be the key of a dict (for fn behaviors_to_params)
    np_behaviors = np.full((max_agents), behavior)
    params = jnp.array([behavior_to_params(behavior) for behavior in np_behaviors])
    behaviors = jnp.array(np_behaviors)
    return AgentState(
        # idx in the entities (ent_idx) state to map agents information in the different data structures
        ent_idx=jnp.arange(max_agents, dtype=int),
        prox=jnp.zeros((max_agents, 2)),
        motor=jnp.zeros((max_agents, 2)),
        behavior=behaviors,
        params=params,
        wheel_diameter=jnp.full((max_agents), wheel_diameter),
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=jnp.full((max_agents), prox_dist_max),
        proxs_cos_min=jnp.full((max_agents), prox_cos_min),
        proximity_map_dist=jnp.zeros((max_agents, 1)),
        proximity_map_theta=jnp.zeros((max_agents, 1)),
        color=jnp.tile(agents_color, (max_agents, 1)),
    )


def init_objects(
    max_agents=MAX_AGENTS, max_objects=MAX_OBJECTS, objects_color=OBJECTS_COLOR
):
    # Entities idx of objects
    start_idx, stop_idx = max_agents, max_agents + max_objects
    objects_ent_idx = jnp.arange(start_idx, stop_idx, dtype=int)

    return ObjectState(
        ent_idx=objects_ent_idx, color=jnp.tile(objects_color, (max_objects, 1))
    )


def init_complete_state(
    entities=None,
    agents=None,
    objects=None,
    box_size=BOX_SIZE,
    max_agents=MAX_AGENTS,
    max_objects=MAX_OBJECTS,
    neighbor_radius=NEIGHBOR_RADIUS,
    collision_alpha=COLLISION_ALPHA,
    collision_eps=COLLISION_EPS,
    dt=DT,
):
    return State(
        time=0,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt,
        entities=entities,
        agents=agents,
        objects=objects,
    )
