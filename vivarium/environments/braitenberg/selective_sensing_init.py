from enum import Enum
from collections import defaultdict

import numpy as np
import jax.numpy as jnp
import matplotlib.colors as mcolors

from jax import random
from jax_md.rigid_body import RigidBody

from omegaconf import OmegaConf
from hydra import initialize, compose

from vivarium.environments.braitenberg.simple import Behaviors, behavior_to_params
from vivarium.environments.braitenberg.selective_sensing_classes import State, AgentState, ObjectState, EntityState, EntityType


# load default arguments of functions from Hydra config files
def load_default_config():
    with initialize(config_path="../../../conf", version_base=None):
        cfg = compose(config_name="config")
        args = OmegaConf.merge(cfg.default, cfg.scene)
    return args

config = load_default_config()


### Helper functions to generate elements of sub states

# Helper function to transform a color string into rgb with matplotlib colors
def _string_to_rgb_array(color_str):
    return jnp.array(list(mcolors.to_rgb(color_str)))

# Helper functions to define behaviors of agents in selecting sensing case
def define_behavior_map(behavior, sensed_mask):
    """Create a dict with behavior value, params and sensed mask for a given behavior

    :param behavior: behavior value
    :param sensed_mask: list of sensed mask
    :return: params associated to the behavior
    """
    params = behavior_to_params(behavior)
    sensed_mask = jnp.array([sensed_mask])

    behavior_map = {
        'behavior': behavior,
        'params': params,
        'sensed_mask': sensed_mask
    }
    return behavior_map

def stack_behaviors(behaviors_dict_list):
    """Return a dict with the stacked information from different behaviors, params and sensed mask

    :param behaviors_dict_list: list of dicts containing behavior, params and sensed mask for 1 behavior
    :return: stacked_behavior_map
    """
    # init variables
    n_behaviors = len(behaviors_dict_list)
    sensed_length = behaviors_dict_list[0]['sensed_mask'].shape[1]

    params = np.zeros((n_behaviors, 2, 3)) # (2, 3) = params.shape
    sensed_mask = np.zeros((n_behaviors, sensed_length))
    behaviors = np.zeros((n_behaviors,))

    # iterate in the list of behaviors and update params and mask
    for i in range(n_behaviors):
        assert behaviors_dict_list[i]['sensed_mask'].shape[1] == sensed_length
        params[i] = behaviors_dict_list[i]['params']
        sensed_mask[i] = behaviors_dict_list[i]['sensed_mask']
        behaviors[i] = behaviors_dict_list[i]['behavior']

    stacked_behavior_map = {
        'behaviors': behaviors,
        'params': params,
        'sensed_mask': sensed_mask
    }

    return stacked_behavior_map

def get_agents_params_and_sensed_arr(agents_stacked_behaviors_list):
    """Generate the behaviors, params and sensed arrays in jax from a list of stacked behaviors

    :param agents_stacked_behaviors_list: list of stacked behaviors
    :return: params, sensed, behaviors
    """
    n_agents = len(agents_stacked_behaviors_list)
    params_shape = agents_stacked_behaviors_list[0]['params'].shape
    sensed_shape = agents_stacked_behaviors_list[0]['sensed_mask'].shape
    behaviors_shape = agents_stacked_behaviors_list[0]['behaviors'].shape
    # Init arrays w right shapes
    params = np.zeros((n_agents, *params_shape))
    sensed = np.zeros((n_agents, *sensed_shape))
    behaviors = np.zeros((n_agents, *behaviors_shape))

    for i in range(n_agents):
        assert agents_stacked_behaviors_list[i]['params'].shape == params_shape
        assert agents_stacked_behaviors_list[i]['sensed_mask'].shape == sensed_shape
        assert agents_stacked_behaviors_list[i]['behaviors'].shape == behaviors_shape
        params[i] = agents_stacked_behaviors_list[i]['params']
        sensed[i] = agents_stacked_behaviors_list[i]['sensed_mask']
        behaviors[i] = agents_stacked_behaviors_list[i]['behaviors']

    params = jnp.array(params)
    sensed = jnp.array(sensed)
    behaviors = jnp.array(behaviors)

    return params, sensed, behaviors

def get_positions(positions, n, box_size):
    if positions is None:
        return [None] * n
    assert len(positions) == n, f"The number of positions: {len(positions)} must match the number of entities: {n}"
    for pos in positions:
        assert len(pos) == 2, f"You have to provide position with 2 coordinates, {pos} has {len(pos)}"
        assert (min(pos) > 0 and max(pos) < box_size), f"Coordinates must be floats between 0 and box_size: {box_size}, found coordinates = {pos}"
    return positions

def check_position_redundancies(agents_pos, objects_pos):
    positions = agents_pos + objects_pos
    position_indices = defaultdict(list)

    for idx, position in enumerate(positions):
        if position is not None:
            position_indices[tuple(position)].append(idx)

    redundant_positions = {position: indices for position, indices in position_indices.items() if len(indices) > 1}

    return redundant_positions if (len(redundant_positions) > 0) else False

def get_exists(exists, n):
    if exists is None:
        return [1] * n
    assert isinstance(exists, int) and (exists < n), f"Exists must be an int inferior than {n}, {exists} is not"
    exists = [1] * exists + [None] * (n - exists)
    return exists

def set_to_none_if_all_none(lst):
    if not any(element is not None for element in lst):
        return None
    return lst

### Helper functions to generate elements sub states of the state

def init_entities(
    max_agents,
    max_objects,
    ent_sub_types,
    n_dims=config.n_dims,
    box_size=config.box_size,
    existing_agents=None,
    existing_objects=None,
    mass_center=config.mass_center,
    mass_orientation=config.mass_orientation,
    diameter=config.diameter,
    friction=config.friction,
    agents_pos=None,
    objects_pos=None,
    key_agents_pos=random.PRNGKey(config.seed),
    key_objects_pos=random.PRNGKey(config.seed+1),
    key_orientations=random.PRNGKey(config.seed+2)
):
    """Init the sub entities state (field of state)"""
    n_entities = max_agents + max_objects # we store the entities data in jax arrays of length max_agents + max_objects 
    # Assign random positions to each entity in the environment
    agents_positions = random.uniform(key_agents_pos, (max_agents, n_dims)) * box_size
    objects_positions = random.uniform(key_objects_pos, (max_objects, n_dims)) * box_size

    # TODO cet aprem
    # Replace random positions with predefined ones if they exist:
    if agents_pos:
        defined_pos = jnp.array([p if p is not None else [-1, -1] for p in agents_pos])
        mask = defined_pos[:, 0] != -1
        agents_positions = jnp.where(mask[:, None], defined_pos, agents_positions)

    if objects_pos:
        defined_pos = jnp.array([p if p is not None else [-1, -1] for p in objects_pos])
        mask = defined_pos[:, 0] != -1
        objects_positions = jnp.where(mask[:, None], defined_pos, objects_positions)

    positions = jnp.concatenate((agents_positions, objects_positions))

    # Assign random orientations between 0 and 2*pi to each entity
    orientations = random.uniform(key_orientations, (n_entities,)) * 2 * jnp.pi 

    # Assign types to the entities
    agents_entities = jnp.full(max_agents, EntityType.AGENT.value)
    object_entities = jnp.full(max_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)

    # Define arrays with existing entities
    exists_agents = jnp.ones((max_agents))
    exists_objects = jnp.ones((max_objects))

    if existing_agents is not None:
        mask = jnp.array([e if e is not None else 0 for e in existing_agents])
        exists_agents = jnp.where(mask != 0, 1, 0)

    if existing_objects is not None:
        mask = jnp.array([e if e is not None else 0 for e in existing_objects])
        exists_objects = jnp.where(mask != 0, 1, 0)

    exists = jnp.concatenate((exists_agents, exists_objects), dtype=int)

    # Works because dictionaries are ordered in Python
    ent_subtypes = np.zeros(n_entities)
    cur_idx = 0
    for subtype_id, n_subtype in ent_sub_types.values():
        ent_subtypes[cur_idx:cur_idx+n_subtype] = subtype_id
        cur_idx += n_subtype
    ent_subtypes = jnp.array(ent_subtypes, dtype=int) 

    return EntityState(
        position=RigidBody(center=positions, orientation=orientations),
        momentum=None,
        force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
        mass=RigidBody(center=jnp.full((n_entities, 1), mass_center), orientation=jnp.full((n_entities), mass_orientation)),
        entity_type=entity_types,
        ent_subtype=ent_subtypes,
        entity_idx = jnp.array(list(range(max_agents)) + list(range(max_objects))),
        diameter=jnp.full((n_entities), diameter),
        friction=jnp.full((n_entities), friction),
        exists=exists
    )

def init_agents(
    max_agents,
    max_objects,
    params,
    sensed,
    behaviors,
    agents_color,
    wheel_diameter=config.wheel_diameter,
    speed_mul=config.speed_mul,
    max_speed=config.max_speed,
    theta_mul=config.theta_mul,
    prox_dist_max=config.prox_dist_max,
    prox_cos_min=config.prox_cos_min
):
    """Init the sub agents state (field of state)"""
    return AgentState(
        # idx in the entities (ent_idx) state to map agents information in the different data structures
        ent_idx=jnp.arange(max_agents, dtype=int), 
        prox=jnp.zeros((max_agents, 2), dtype=float),
        prox_sensed_ent=jnp.zeros((max_agents, 2), dtype=int),
        motor=jnp.zeros((max_agents, 2)),
        behavior=behaviors,
        params=params,
        sensed=sensed,
        wheel_diameter=jnp.full((max_agents), wheel_diameter),
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=jnp.full((max_agents), prox_dist_max),
        proxs_cos_min=jnp.full((max_agents), prox_cos_min),
        # Change shape of these maps so they stay constant (jax.lax.scan problem otherwise)
        proximity_map_dist=jnp.zeros((max_agents, max_agents + max_objects)),
        proximity_map_theta=jnp.zeros((max_agents, max_agents + max_objects)),
        color=agents_color
    )

def init_objects(
    max_agents,
    max_objects,
    objects_color
):
    """Init the sub objects state (field of state)"""
    start_idx, stop_idx = max_agents, max_agents + max_objects 
    objects_ent_idx = jnp.arange(start_idx, stop_idx, dtype=int)

    return ObjectState(
        ent_idx=objects_ent_idx,
        color=objects_color
    )


def init_complete_state(
    entities,
    agents,
    objects,
    max_agents,
    max_objects,
    total_ent_sub_types,
    box_size=config.box_size,
    neighbor_radius=config.neighbor_radius,
    collision_alpha=config.collision_alpha,
    collision_eps=config.collision_eps,
    dt=config.dt,
):
    """Init the complete state"""
    return  State(
        time=0,
        dt=dt,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        entities=entities,
        agents=agents,
        objects=objects,
        ent_sub_types=total_ent_sub_types
    )   


def init_state(
    entities_data=config.entities_data,
    box_size=config.box_size,
    dt=config.dt,
    neighbor_radius=config.neighbor_radius,
    collision_alpha=config.collision_alpha,
    collision_eps=config.collision_eps,
    n_dims=config.n_dims,
    seed=config.seed,
    diameter=config.diameter,
    friction=config.friction,
    mass_center=config.mass_center,
    mass_orientation=config.mass_orientation,
    existing_agents=None,
    existing_objects=None,
    wheel_diameter=config.wheel_diameter,
    speed_mul=config.speed_mul,
    max_speed=config.max_speed,
    theta_mul=config.theta_mul,
    prox_dist_max=config.prox_dist_max,
    prox_cos_min=config.prox_cos_min,
) -> State:
    """ Init the jax state of the simulation from classical python / yaml scene arguments """
    key = random.PRNGKey(seed)
    key, key_agents_pos, key_objects_pos, key_orientations = random.split(key, 4)
    
    # create an enum for entities subtypes
    ent_sub_types = entities_data['EntitySubTypes']
    ent_sub_types_enum = Enum('ent_sub_types_enum', {ent_sub_types[i]: i for i in range(len(ent_sub_types))}) 
    ent_data = entities_data['Entities']

    # create max agents and max objects
    max_agents = 0
    max_objects = 0 

    # create agent and objects dictionaries 
    agents_data = {}
    objects_data = {}

    # create agents and objects positions lists
    agents_pos = []
    objects_pos = []
    agents_exist = []
    objects_exist = []

    # iterate over the entities subtypes
    for ent_sub_type in ent_sub_types:
        # get their data in the ent_data
        data = ent_data[ent_sub_type]
        color_str = data['color']
        color = _string_to_rgb_array(color_str)
        n = data['num']
        
        # Check if the entity is an agent or an object
        if data['type'] == 'AGENT':
            max_agents += n
            behavior_list = []
            # create a behavior list for all behaviors of the agent
            for beh_name, behavior_data in data['selective_behaviors'].items():
                beh_name = behavior_data['beh']
                behavior_id = Behaviors[beh_name].value
                # Init an empty mask
                sensed_mask = np.zeros((len(ent_sub_types, )))
                for sensed_type in behavior_data['sensed']:
                    try:
                        # Iteratively update it with specific sensed values
                        sensed_id = ent_sub_types_enum[sensed_type].value
                        sensed_mask[sensed_id] = 1
                    except KeyError:
                        raise ValueError(f"Unknown sensed_type '{sensed_type}' encountered in sensed entities for {ent_sub_type}. Please select entities among {ent_sub_types}")
                beh = define_behavior_map(behavior_id, sensed_mask)
                behavior_list.append(beh)
            # stack the elements of the behavior list and update the agents_data dictionary
            stacked_behaviors = stack_behaviors(behavior_list)
            agents_data[ent_sub_type] = {'n': n, 'color': color, 'stacked_behs': stacked_behaviors}
            positions = get_positions(data.get('positions'), n, box_size)
            agents_pos.extend(positions)
            exists = get_exists(data.get('existing'), n)
            agents_exist.extend(exists)

        # only updated object counters and color if entity is an object
        elif data['type'] == 'OBJECT':
            max_objects += n
            objects_data[ent_sub_type] = {'n': n, 'color': color}
            positions = get_positions(data.get('positions'), n, box_size)
            objects_pos.extend(positions)
            exists = get_exists(data.get('existing'), n)
            objects_exist.extend(exists)

    # Check for redundant positions
    redundant_positions = check_position_redundancies(agents_pos, objects_pos)
    if redundant_positions:
        redundant_positions_list = list(redundant_positions.keys())
        raise ValueError(f"Impossible to initialize the simulation state with redundant positions : {redundant_positions_list}. This would lead to collision errors in the physics engine of the environment")
    # Set positions to None lists if they don't contain any positions 
    agents_pos = set_to_none_if_all_none(agents_pos)
    objects_pos = set_to_none_if_all_none(objects_pos)
    agents_exist = set_to_none_if_all_none(agents_exist)
    objects_exist = set_to_none_if_all_none(objects_exist)

    # Create the params, sensed, behaviors and colors arrays 
    ag_colors_list = []
    agents_stacked_behaviors_list = []
    total_ent_sub_types = {}
    # iterate over agent types
    for agent_type, data in agents_data.items():
        n = data['n']
        stacked_behavior = data['stacked_behs']
        n_stacked_behavior = list([stacked_behavior] * n)
        tiled_color = list(np.tile(data['color'], (n, 1)))
        # update the lists with behaviors and color elements
        agents_stacked_behaviors_list = agents_stacked_behaviors_list + n_stacked_behavior
        ag_colors_list = ag_colors_list + tiled_color
        total_ent_sub_types[agent_type] = (ent_sub_types_enum[agent_type].value, n)

    # create the final jnp arrays
    agents_colors = jnp.concatenate(jnp.array([ag_colors_list]), axis=0)
    params, sensed, behaviors = get_agents_params_and_sensed_arr(agents_stacked_behaviors_list)

    # do the same for objects colors
    obj_colors_list = []
    # iterate over object types
    for objecy_type, data in objects_data.items():
        n = data['n']
        tiled_color = list(np.tile(data['color'], (n, 1)))
        obj_colors_list = obj_colors_list + tiled_color
        total_ent_sub_types[objecy_type] = (ent_sub_types_enum[objecy_type].value, n)

    objects_colors = jnp.concatenate(jnp.array([obj_colors_list]), axis=0)
    # print(total_ent_sub_types)

    # Init sub states and total state
    entities = init_entities(
        max_agents=max_agents,
        max_objects=max_objects,
        ent_sub_types=total_ent_sub_types,
        n_dims=n_dims,
        box_size=box_size,
        existing_agents=agents_exist,
        existing_objects=objects_exist,
        mass_center=mass_center,
        mass_orientation=mass_orientation,
        diameter=diameter,
        friction=friction,
        agents_pos=agents_pos,
        objects_pos=objects_pos,
        key_agents_pos=key_agents_pos,
        key_objects_pos=key_objects_pos,
        key_orientations=key_orientations
    )

    agents = init_agents(
        max_agents=max_agents,
        max_objects=max_objects,
        params=params,
        sensed=sensed,
        behaviors=behaviors,
        agents_color=agents_colors,
        wheel_diameter=wheel_diameter,
        speed_mul=speed_mul,
        max_speed=max_speed,
        theta_mul=theta_mul,
        prox_dist_max=prox_dist_max,
        prox_cos_min=prox_cos_min
    )

    objects = init_objects(
        max_agents=max_agents,
        max_objects=max_objects,
        objects_color=objects_colors
    )

    state = init_complete_state(
        entities=entities,
        agents=agents,
        objects=objects,
        max_agents=max_agents,
        max_objects=max_objects,
        total_ent_sub_types=total_ent_sub_types,
        box_size=box_size,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt
    )

    return state

