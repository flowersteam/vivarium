from jax_md.rigid_body import RigidBody

import simulator_pb2

from vivarium.simulator.grpc_server.numproto.numproto import (
    proto_to_ndarray,
    ndarray_to_proto,
)
from vivarium.simulator.simulator_states import (
    SimulatorState,
    EntityState,
    AgentState,
    ObjectState,
)
from vivarium.simulator.simulator_states import SimState as State


def proto_to_state(state):
    """Convert a protobuf state to a State object.

    :param state: simulation state in protobuf format
    :return: State object
    """
    return State(
        simulator_state=proto_to_simulator_state(state.simulator_state),
        entity_state=proto_to_nve_state(state.entity_state),
        agent_state=proto_to_agent_state(state.agent_state),
        object_state=proto_to_object_state(state.object_state),
    )


# Added time, sensed, params and entity subtypes
def proto_to_simulator_state(simulator_state):
    """Convert a protobuf simulator state to a SimulatorState object.

    :param simulator_state: simulator state in protobuf format
    :return: SimulatorState object
    """
    return SimulatorState(
        idx=proto_to_ndarray(simulator_state.idx).astype(int),
        box_size=proto_to_ndarray(simulator_state.box_size).astype(float),
        time=proto_to_ndarray(simulator_state.time).astype(int),
        max_agents=proto_to_ndarray(simulator_state.max_agents).astype(int),
        max_objects=proto_to_ndarray(simulator_state.max_objects).astype(int),
        num_steps_lax=proto_to_ndarray(simulator_state.num_steps_lax).astype(int),
        dt=proto_to_ndarray(simulator_state.dt).astype(float),
        freq=proto_to_ndarray(simulator_state.freq).astype(float),
        neighbor_radius=proto_to_ndarray(simulator_state.neighbor_radius).astype(float),
        to_jit=proto_to_ndarray(simulator_state.to_jit).astype(int),
        use_fori_loop=proto_to_ndarray(simulator_state.use_fori_loop).astype(int),
        collision_eps=proto_to_ndarray(simulator_state.collision_eps).astype(float),
        collision_alpha=proto_to_ndarray(simulator_state.collision_alpha).astype(float),
    )


def proto_to_nve_state(entity_state):
    """Convert a protobuf entity state to an EntityState object.

    :param entity_state: entity state in protobuf format
    :return: EntityState object
    """
    return EntityState(
        position=RigidBody(
            center=proto_to_ndarray(entity_state.position.center).astype(float),
            orientation=proto_to_ndarray(entity_state.position.orientation).astype(
                float
            ),
        ),
        momentum=RigidBody(
            center=proto_to_ndarray(entity_state.momentum.center).astype(float),
            orientation=proto_to_ndarray(entity_state.momentum.orientation).astype(
                float
            ),
        ),
        force=RigidBody(
            center=proto_to_ndarray(entity_state.force.center).astype(float),
            orientation=proto_to_ndarray(entity_state.force.orientation).astype(float),
        ),
        mass=RigidBody(
            center=proto_to_ndarray(entity_state.mass.center).astype(float),
            orientation=proto_to_ndarray(entity_state.mass.orientation).astype(float),
        ),
        entity_type=proto_to_ndarray(entity_state.entity_type).astype(int),
        ent_subtype=proto_to_ndarray(entity_state.ent_subtype).astype(int),
        entity_idx=proto_to_ndarray(entity_state.entity_idx).astype(int),
        diameter=proto_to_ndarray(entity_state.diameter).astype(float),
        friction=proto_to_ndarray(entity_state.friction).astype(float),
        exists=proto_to_ndarray(entity_state.exists).astype(int),
    )


def proto_to_agent_state(agent_state):
    """Convert a protobuf agent state to an AgentState object.

    :param agent_state: agent state in protobuf format
    :return: AgentState object
    """
    return AgentState(
        ent_idx=proto_to_ndarray(agent_state.ent_idx).astype(int),
        proximity_map_dist=proto_to_ndarray(agent_state.proximity_map_dist).astype(
            float
        ),
        proximity_map_theta=proto_to_ndarray(agent_state.proximity_map_theta).astype(
            float
        ),
        prox=proto_to_ndarray(agent_state.prox).astype(float),
        prox_sensed_ent_type=proto_to_ndarray(agent_state.prox_sensed_ent_type).astype(
            int
        ),
        prox_sensed_ent_idx=proto_to_ndarray(agent_state.prox_sensed_ent_idx).astype(
            int
        ),
        motor=proto_to_ndarray(agent_state.motor).astype(float),
        behavior=proto_to_ndarray(agent_state.behavior).astype(int),
        params=proto_to_ndarray(agent_state.params).astype(float),
        sensed=proto_to_ndarray(agent_state.sensed).astype(float),
        wheel_diameter=proto_to_ndarray(agent_state.wheel_diameter).astype(float),
        speed_mul=proto_to_ndarray(agent_state.speed_mul).astype(float),
        max_speed=proto_to_ndarray(agent_state.max_speed).astype(float),
        theta_mul=proto_to_ndarray(agent_state.theta_mul).astype(float),
        proxs_dist_max=proto_to_ndarray(agent_state.proxs_dist_max).astype(float),
        proxs_cos_min=proto_to_ndarray(agent_state.proxs_cos_min).astype(float),
        color=proto_to_ndarray(agent_state.color).astype(float),
    )


def proto_to_object_state(object_state):
    """Convert a protobuf object state to an ObjectState object.

    :param object_state: object state in protobuf format
    :return: ObjectState object
    """
    return ObjectState(
        ent_idx=proto_to_ndarray(object_state.ent_idx).astype(int),
        color=proto_to_ndarray(object_state.color).astype(float),
    )


def state_to_proto(state):
    """Convert a State object to a protobuf state.

    :param state: simulation state
    :return: protobuf state
    """
    return simulator_pb2.State(
        simulator_state=simulator_state_to_proto(state.simulator_state),
        entity_state=nve_state_to_proto(state.entity_state),
        agent_state=agent_state_to_proto(state.agent_state),
        object_state=object_state_to_proto(state.object_state),
    )


def simulator_state_to_proto(simulator_state):
    """Convert a SimulatorState object to a protobuf simulator state.

    :param simulator_state: SimulatorState object
    :return: protobuf simulator state
    """
    return simulator_pb2.SimulatorState(
        idx=ndarray_to_proto(simulator_state.idx),
        box_size=ndarray_to_proto(simulator_state.box_size),
        time=ndarray_to_proto(simulator_state.time),
        max_agents=ndarray_to_proto(simulator_state.max_agents),
        max_objects=ndarray_to_proto(simulator_state.max_objects),
        num_steps_lax=ndarray_to_proto(simulator_state.num_steps_lax),
        dt=ndarray_to_proto(simulator_state.dt),
        freq=ndarray_to_proto(simulator_state.freq),
        neighbor_radius=ndarray_to_proto(simulator_state.neighbor_radius),
        to_jit=ndarray_to_proto(simulator_state.to_jit),
        use_fori_loop=ndarray_to_proto(simulator_state.use_fori_loop),
        collision_eps=ndarray_to_proto(simulator_state.collision_eps),
        collision_alpha=ndarray_to_proto(simulator_state.collision_alpha),
    )


def nve_state_to_proto(entity_state):
    """Convert an EntityState object to a protobuf entity state.

    :param entity_state: EntityState object
    :return: protobuf entity state
    """
    return simulator_pb2.EntityState(
        position=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.position.center),
            orientation=ndarray_to_proto(entity_state.position.orientation),
        ),
        momentum=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.momentum.center),
            orientation=ndarray_to_proto(entity_state.momentum.orientation),
        ),
        force=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.force.center),
            orientation=ndarray_to_proto(entity_state.force.orientation),
        ),
        mass=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.mass.center),
            orientation=ndarray_to_proto(entity_state.mass.orientation),
        ),
        entity_type=ndarray_to_proto(entity_state.entity_type),
        ent_subtype=ndarray_to_proto(entity_state.ent_subtype),
        entity_idx=ndarray_to_proto(entity_state.entity_idx),
        diameter=ndarray_to_proto(entity_state.diameter),
        friction=ndarray_to_proto(entity_state.friction),
        exists=ndarray_to_proto(entity_state.exists),
    )


def agent_state_to_proto(agent_state):
    """Convert an AgentState object to a protobuf agent state.

    :param agent_state: AgentState object
    :return: protobuf agent state
    """
    return simulator_pb2.AgentState(
        ent_idx=ndarray_to_proto(agent_state.ent_idx),
        proximity_map_dist=ndarray_to_proto(agent_state.proximity_map_dist),
        proximity_map_theta=ndarray_to_proto(agent_state.proximity_map_theta),
        prox=ndarray_to_proto(agent_state.prox),
        prox_sensed_ent_type=ndarray_to_proto(agent_state.prox_sensed_ent_type),
        prox_sensed_ent_idx=ndarray_to_proto(agent_state.prox_sensed_ent_idx),
        motor=ndarray_to_proto(agent_state.motor),
        behavior=ndarray_to_proto(agent_state.behavior),
        sensed=ndarray_to_proto(agent_state.sensed),
        params=ndarray_to_proto(agent_state.params),
        wheel_diameter=ndarray_to_proto(agent_state.wheel_diameter),
        speed_mul=ndarray_to_proto(agent_state.speed_mul),
        max_speed=ndarray_to_proto(agent_state.max_speed),
        theta_mul=ndarray_to_proto(agent_state.theta_mul),
        proxs_dist_max=ndarray_to_proto(agent_state.proxs_dist_max),
        proxs_cos_min=ndarray_to_proto(agent_state.proxs_cos_min),
        color=ndarray_to_proto(agent_state.color),
    )


def object_state_to_proto(object_state):
    """Convert an ObjectState object to a protobuf object state.

    :param object_state: ObjectState object
    :return: protobuf object state
    """
    return simulator_pb2.ObjectState(
        ent_idx=ndarray_to_proto(object_state.ent_idx),
        color=ndarray_to_proto(object_state.color),
    )
