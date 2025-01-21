import logging as lg

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from jax import vmap, jit
from jax import random, lax
from jax_md import space, partition

from vivarium.environments.utils import distance
from vivarium.environments.base_env import BaseEnv
from vivarium.environments.physics_engine import dynamics_fn
from vivarium.environments.braitenberg.behaviors import Behaviors
from vivarium.environments.braitenberg.selective_sensing.classes import (
    State,
    Neighbors,
    EntityType,
)
from vivarium.environments.braitenberg.selective_sensing.init import init_state

from vivarium.environments.braitenberg.simple.simple_env import (
    proximity_map,
    sensor_fn,
    linear_behavior,
    braintenberg_force_fn,
)


SPACE_NDIMS = 2


# TODO : Should refactor the function to split the returns
def get_relative_displacement(state, agents_neighs_idx, displacement_fn):
    """Get all infos relative to distance and orientation between all agents and their neighbors

    :param state: state
    :param agents_neighs_idx: idx all agents neighbors
    :param displacement_fn: jax md function enabling to know the distance between points
    :return: distance array, angles array, distance map for all agents, angles map for all agents
    """
    body = state.entities.position
    senders, receivers = agents_neighs_idx
    Ra = body.center[senders]
    Rb = body.center[receivers]
    dR = -space.map_bond(displacement_fn)(
        Ra, Rb
    )  # Looks like it should be opposite, but don't understand why

    dist, theta = proximity_map(dR, body.orientation[senders])
    proximity_map_dist = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_dist = proximity_map_dist.at[senders, receivers].set(dist)
    proximity_map_theta = jnp.zeros(
        (state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0])
    )
    proximity_map_theta = proximity_map_theta.at[senders, receivers].set(theta)
    return dist, theta, proximity_map_dist, proximity_map_theta


def compute_motor(proxs, params, behaviors, motors):
    """Compute new motor values. If behavior is manual, keep same motor values. Else, compute new values with proximeters and params.

    :param proxs: proximeters of all agents
    :param params: parameters mapping proximeters to new motor values
    :param behaviors: array of behaviors
    :param motors: current motor values
    :return: new motor values
    """
    manual = jnp.where(behaviors == Behaviors.MANUAL.value, 1, 0)
    manual_mask = manual
    linear_motor_values = linear_behavior(proxs, params)
    motor_values = linear_motor_values * (1 - manual_mask) + motors * manual_mask
    return motor_values


### 1 : Functions for selective sensing with occlusion


def update_mask(mask, left_n_right_types, ent_type):
    """Update a mask of

    :param mask: mask that will be applied on sensors of agents
    :param left_n_right_types: types of left adn right sensed entities
    :param ent_type: entity subtype (e.g 1 for predators)
    :return: mask
    """
    cur = jnp.where(left_n_right_types == ent_type, 0, 1)
    mask *= cur
    return mask


def keep_mask(mask, left_n_right_types, ent_type):
    """Return the mask unchanged

    :param mask: mask
    :param left_n_right_types: left_n_right_types
    :param ent_type: ent_type
    :return: mask
    """
    return mask


def mask_proxs_occlusion(proxs, left_n_right_types, ent_sensed_arr):
    """Mask the proximeters of agents with occlusion

    :param proxs: proxiemters of agents without occlusion (shape = (2,))
    :param e_sensed_types: types of both entities sensed at left and right (shape=(2,))
    :param ent_sensed_arr: mask of sensed subtypes by the agent (e.g jnp.array([0, 1, 0, 1]) if sense only entities of subtype 1 and 4)
    :return: updated proximeters according to sensed_subtypes
    """
    mask = jnp.array([1, 1])
    # Iterate on the array of sensed entities mask
    for ent_type, sensed in enumerate(ent_sensed_arr):
        # If an entity is sensed, update the mask, else keep it as it is
        mask = jax.lax.cond(
            sensed, update_mask, keep_mask, mask, left_n_right_types, ent_type
        )
    # Update the mask with 0s where the mask is, else keep the prox value
    proxs = jnp.where(mask, 0, proxs)
    return proxs


# Example :
# ent_sensed_arr = jnp.array([0, 1, 0, 0, 1])
# proxs = jnp.array([0.8, 0.2])
# e_sensed_types = jnp.array([4, 4]) # Modify these values to check it works
# print(mask_proxs_occlusion(proxs, e_sensed_types, ent_sensed_arr))


def compute_behavior_motors(
    state, params, sensed_mask, behavior, motor, agent_proxs, sensed_ent_idx
):
    """Compute the motor values for a specific behavior

    :param state: state
    :param params: behavior params params
    :param sensed_mask: sensed_mask for this behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_proxs: agent proximeters (unmasked)
    :param sensed_ent_idx: idx of left and right entities sensed
    :return: right motor values for this behavior
    """
    left_n_right_types = state.entities.ent_subtype[sensed_ent_idx]
    behavior_proxs = mask_proxs_occlusion(agent_proxs, left_n_right_types, sensed_mask)
    motors = compute_motor(behavior_proxs, params, behaviors=behavior, motors=motor)
    return motors


# See for the vectorizing idx because already in a vmaped function here
compute_all_behavior_motors = vmap(
    compute_behavior_motors, in_axes=(None, 0, 0, 0, None, None, None)
)


def compute_occlusion_proxs_motors(
    state,
    agent_idx,
    params,
    sensed,
    behaviors,
    motor,
    raw_proxs,
    ag_idx_dense_senders,
    ag_idx_dense_receivers,
):
    """_summary_

    :param state: state
    :param agent_idx: agent idx in entities
    :param params: params arrays for all agent's behaviors
    :param sensed: sensed mask arrays for all agent's behaviors
    :param behaviors: agent behaviors array
    :param motor: agent motors
    :param raw_proxs: raw_proximeters for all agents (shape=(n_agents * (n_entities - 1), 2))
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: _description_
    """
    behavior = jnp.expand_dims(behaviors, axis=1)
    # Compute the neighbors idx of the agent and get its raw proximeters (of shape (n_entities -1 , 2))
    ent_ag_neighs_idx = ag_idx_dense_senders[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_neighs_idx]

    # Get the max and arg max of these proximeters on axis 0, gives results of shape (2,)
    agent_proxs = jnp.max(agent_raw_proxs, axis=0)
    argmax = jnp.argmax(agent_raw_proxs, axis=0)
    # Get the real entity idx of the left and right sensed entities from dense neighborhoods
    sensed_ent_idx = ag_idx_dense_receivers[agent_idx][argmax]
    prox_sensed_ent_types = state.entities.ent_subtype[sensed_ent_idx]

    # Compute the motor values for all behaviors and do a mean on it
    motor_values = compute_all_behavior_motors(
        state, params, sensed, behavior, motor, agent_proxs, sensed_ent_idx
    )
    motors = jnp.mean(motor_values, axis=0)

    return agent_proxs, (sensed_ent_idx, prox_sensed_ent_types), motors


compute_all_agents_proxs_motors_occl = vmap(
    compute_occlusion_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None)
)


### 2 : Functions for selective sensing without occlusion


def mask_sensors(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Mask the raw proximeters of agents for a specific entity type

    :param state: state
    :param agent_raw_proxs: raw_proximeters of agent (shape=(n_entities - 1), 2)
    :param ent_type_id: entity subtype id (e.g 0 for PREYS)
    :param ent_neighbors_idx: idx of agent neighbors in entities arrays
    :return: updated agent raw proximeters
    """
    mask = jnp.where(state.entities.ent_subtype[ent_neighbors_idx] == ent_type_id, 0, 1)
    mask = jnp.expand_dims(mask, 1)
    mask = jnp.broadcast_to(mask, agent_raw_proxs.shape)
    return agent_raw_proxs * mask


def dont_change(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Leave the agent raw_proximeters unchanged

    :param state: state
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_type_id: ent_type_id
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: agent_raw_proxs
    """
    return agent_raw_proxs


def compute_behavior_prox(state, agent_raw_proxs, ent_neighbors_idx, sensed_entities):
    """Compute the proximeters for a specific behavior

    :param state: state
    :param agent_raw_proxs: agent raw proximeters
    :param ent_neighbors_idx: idx of agent neighbors
    :param sensed_entities: array of sensed entities
    :return: updated proximeters
    """
    # iterate over all the types in sensed_entities and return if they are sensed or not
    for ent_type_id, sensed in enumerate(sensed_entities):
        # change the proxs if you don't perceive the entity, else leave them unchanged
        agent_raw_proxs = lax.cond(
            sensed,
            dont_change,
            mask_sensors,
            state,
            agent_raw_proxs,
            ent_type_id,
            ent_neighbors_idx,
        )
    # Compute the final proxs with a max on the updated raw_proxs
    proxs = jnp.max(agent_raw_proxs, axis=0)
    return proxs


def compute_behavior_proxs_motors(
    state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx
):
    """Return the proximeters and the motors for a specific behavior

    :param state: state
    :param params: params of the behavior
    :param sensed: sensed mask of the behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: behavior proximeters, behavior motors
    """
    behavior_prox = compute_behavior_prox(
        state, agent_raw_proxs, ent_neighbors_idx, sensed
    )
    behavior_motors = compute_motor(behavior_prox, params, behavior, motor)
    return behavior_prox, behavior_motors


# vmap on params, sensed and behavior (parallelize on all agents behaviors at once, but not motorrs because are the same)
compute_all_behavior_proxs_motors = vmap(
    compute_behavior_proxs_motors, in_axes=(None, 0, 0, 0, None, None, None)
)


def compute_agent_proxs_motors(
    state,
    agent_idx,
    params,
    sensed,
    behavior,
    motor,
    raw_proxs,
    ag_idx_dense_senders,
    ag_idx_dense_receivers,
):
    """Compute the agent proximeters and motors for all behaviors

    :param state: state
    :param agent_idx: idx of the agent in entities
    :param params: array of params for all behaviors
    :param sensed: array of sensed mask for all behaviors
    :param behavior: array of behaviors
    :param motor: motor values
    :param raw_proxs: raw_proximeters of all agents
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: array of agent_proximeters, mean of behavior motors
    """
    behavior = jnp.expand_dims(behavior, axis=1)
    ent_ag_idx = ag_idx_dense_senders[agent_idx]
    ent_neighbors_idx = ag_idx_dense_receivers[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_idx]

    # vmap on params, sensed, behaviors and motorss (vmap on all agents)
    agent_proxs, agent_motors = compute_all_behavior_proxs_motors(
        state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx
    )
    mean_agent_motors = jnp.mean(agent_motors, axis=0)

    # need to return a dummy array as 2nd argument to match the compute_agent_proxs_motors function returns with occlusion
    dummy = (jnp.zeros(1), jnp.zeros(1))
    return agent_proxs, dummy, mean_agent_motors


compute_all_agents_proxs_motors = vmap(
    compute_agent_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None)
)


# TODO : Fix the non occlusion error in the step
class SelectiveSensorsEnv(BaseEnv):
    def __init__(self, state, occlusion=True, seed=42):
        """Init the selective sensors braitenberg env

        :param state: simulation state already complete
        :param occlusion: wether to use sensors with occlusion or not, defaults to True
        :param seed: random seed, defaults to 42
        """
        self.seed = seed
        self.occlusion = occlusion
        self.compute_all_agents_proxs_motors = self.choose_agent_prox_motor_function()
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(
            self.displacement, self.shift, braintenberg_force_fn
        )
        # Do a warning at the moment if neighbor radius is < box_size
        if state.neighbor_radius < state.box_size:
            lg.warn(
                "Neighbor radius < Box size, this might cause problems for neighbors arrays and proximity maps updates"
            )
        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            state.box_size,
            r_cutoff=state.neighbor_radius,
            dr_threshold=10.0,
            capacity_multiplier=1.5,
            format=partition.Sparse,
        )
        self.neighbors_storage = self.allocate_neighbors(state)

    def distance(self, point1, point2):
        """Returns the distance between two points

        :param point1: point1 coordinates
        :param point2: point1 coordinates
        :return: distance between two points
        """
        return distance(self.displacement, point1, point2)

    # At the moment doesn't work because the _step function isn't recompiled
    def choose_agent_prox_motor_function(self):
        """Returns the function to compute the proximeters and the motors with or without occlusion

        :return: compute_all_agents_proxs_motors function
        """
        if self.occlusion:
            prox_motor_function = compute_all_agents_proxs_motors_occl
        else:
            prox_motor_function = compute_all_agents_proxs_motors
        return prox_motor_function

    @partial(jit, static_argnums=(0,))
    def _step_env(
        self, state: State, neighbors_storage: Neighbors
    ) -> Tuple[State, Neighbors]:
        """Do one jitted step in the environment and return the updated state, as well as updated neighbors array

        :param state: current state
        :param neighbors_storage: class storing all neighbors information
        :return: new state, neighbors storage wih updated neighbors
        """

        # Retrieve different neighbors format
        neighbors = neighbors_storage.neighbors
        agents_neighs_idx = neighbors_storage.agents_neighs_idx
        ag_idx_dense = neighbors_storage.agents_idx_dense
        senders, receivers = agents_neighs_idx
        ag_idx_dense_senders, ag_idx_dense_receivers = ag_idx_dense

        # Compute raw proxs for all agents first
        dist, relative_theta, proximity_dist_map, proximity_dist_theta = (
            get_relative_displacement(
                state, agents_neighs_idx, displacement_fn=self.displacement
            )
        )

        dist_max = state.agents.proxs_dist_max[senders]
        cos_min = state.agents.proxs_cos_min[senders]
        # changed agents_neighs_idx[1, :] to receivers in line below (check if it works)
        target_exist_mask = state.entities.exists[receivers]
        # Compute agents raw proximeters (proximeters for all neighbors)
        raw_proxs = sensor_fn(
            dist, relative_theta, dist_max, cos_min, target_exist_mask
        )

        # Compute real agents proximeters and motors
        agent_proxs, prox_sensed_ent_tuple, mean_agent_motors = (
            self.compute_all_agents_proxs_motors(
                state,
                state.agents.ent_idx,
                state.agents.params,
                state.agents.sensed,
                state.agents.behavior,
                state.agents.motor,
                raw_proxs,
                ag_idx_dense_senders,
                ag_idx_dense_receivers,
            )
        )

        prox_sensed_ent_idx, prox_sensed_ent_type = prox_sensed_ent_tuple

        # Update agents state
        agents = state.agents.set(
            prox=agent_proxs,
            prox_sensed_ent_type=prox_sensed_ent_type,
            prox_sensed_ent_idx=prox_sensed_ent_idx,
            proximity_map_dist=proximity_dist_map,
            proximity_map_theta=proximity_dist_theta,
            motor=mean_agent_motors,
        )

        # Update the entities and the state
        state = state.set(agents=agents)
        entities = self.apply_physics(state, neighbors)
        state = state.set(time=state.time + 1, entities=entities)

        # Update the neighbors storage
        neighbors = neighbors.update(state.entities.position.center)
        neighbors_storage = Neighbors(
            neighbors=neighbors,
            agents_neighs_idx=agents_neighs_idx,
            agents_idx_dense=ag_idx_dense,
        )

        return state, neighbors_storage

    @partial(jax.jit, static_argnums=(0, 3))
    def _steps(self, state, neighbor_storage, num_updates):
        lg.debug("Compile _steps function in SelectiveSensing environment")

        """Update the current state by doing a _step_env update num_updates times (this results in faster simulations) 

        :param state: _description_
        :param neighbor_storage: _description_
        :param num_updates: _description_
        """

        def step_fn(carry, _):
            """Apply a step function to return new state and neighbors storage in a jax.lax.scan update

            :param carry: tuple of (state, neighbors storage)
            :param _: dummy xs for jax.lax.scan
            :return: tuple of (carry, carry) with carry=(new_state, new_neighbors _sotrage)
            """
            state, neighbors_storage = carry
            new_state, new_neighbors_storage = self._step_env(state, neighbors_storage)
            carry = (new_state, new_neighbors_storage)
            return carry, carry

        (state, neighbor_storage), _ = jax.lax.scan(
            step_fn, (state, neighbor_storage), xs=None, length=num_updates
        )

        return state, neighbor_storage

    def step(self, state: State, num_updates: int = 4) -> State:
        """Do num_updates jitted steps in the environment and return the updated state. This function also handles the neighbors mechanism and hence isn't jitted

        :param state: current state
        :param num_updates: number of jitted_steps
        :return: next state
        """
        # Because momentum is initialized to None, need to initialize it with init_fn from jax_md
        if state.entities.momentum is None:
            state = self.init_fn(state, self.init_key)
            state, neighbors_storage = self._step_env(state, self.neighbors_storage)

        # Save the first state
        current_state = state

        state, neighbors_storage = self._steps(
            current_state, self.neighbors_storage, num_updates
        )

        # Check if neighbors buffer overflowed
        if neighbors_storage.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state if it is the case
            lg.warning(
                f"NEIGHBORS BUFFER OVERFLOW at step {state.time}: rebuilding neighbors"
            )
            self.neighbors_storage = self.allocate_neighbors(state)
            # Because there was an error, we need to re-run this simulation loop from the copy of the current_state we created (and check wether it worked or not after)
            state, neighbors_storage = self._steps(
                current_state, self.neighbors_storage, num_updates
            )
            assert not neighbors_storage.neighbors.did_buffer_overflow

        return state

    def allocate_neighbors(self, state, position=None):
        """Allocate the neighbors according to the state

        :param state: state
        :param position: position of entities in the state, defaults to None
        :return: Neighbors object with neighbors (sparse representation), idx of agent's neighbors, neighbors (dense representation)
        """
        # get the sparse representation of neighbors (shape=(n_neighbors_pairs, 2))
        position = state.entities.position.center if position is None else position
        neighbors = self.neighbor_fn.allocate(position)

        # Also update the neighbor idx of agents
        ag_idx = state.entities.entity_type[neighbors.idx[0]] == EntityType.AGENT.value
        agents_neighs_idx = neighbors.idx[:, ag_idx]

        # Give the idx of the agents in sparse representation, under a dense representation (used to get the raw proxs in compute motors function)
        agents_idx_dense_senders = jnp.array(
            [
                jnp.argwhere(jnp.equal(agents_neighs_idx[0, :], idx)).flatten()
                for idx in jnp.arange(state.max_agents)
            ]
        )
        # Note: jnp.argwhere(jnp.equal(self.agents_neighs_idx[0, :], idx)).flatten() ~ jnp.where(agents_idx[0, :] == idx)

        # Give the idx of the agent neighbors in dense representation
        agents_idx_dense_receivers = agents_neighs_idx[1, :][agents_idx_dense_senders]
        agents_idx_dense = agents_idx_dense_senders, agents_idx_dense_receivers

        neighbor_storage = Neighbors(
            neighbors=neighbors,
            agents_neighs_idx=agents_neighs_idx,
            agents_idx_dense=agents_idx_dense,
        )
        return neighbor_storage


if __name__ == "__main__":
    state = init_state()
    env = SelectiveSensorsEnv(state)

    env.step(state, num_updates=5)
    env.step(state, num_updates=6)
