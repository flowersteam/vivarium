import numpy as np
import jax.numpy as jnp
from jax_md.rigid_body import RigidBody

import simulator_pb2

from vivarium.simulator.grpc_server.numproto.numproto import (
    proto_to_ndarray,
    ndarray_to_proto,
)
from vivarium.simulator.simulator_states import (
    SimulatorState,
    # EntityState,
    AgentState,
    ObjectState,
    SimState
)
from vivarium.environments.braitenberg.selective_sensing.state import EntityState

from vivarium.simulator.simulator_states import SimState as State


from jax_md.dataclasses import fields


def changes_to_proto(changes):
    if isinstance(changes, list) and '__idx' in changes[0]:
        proto_changes = simulator_pb2.Changes()
        for change in changes:
            proto_change = simulator_pb2.Change()
            for k, v in change.items():
                idx_or_value = simulator_pb2.IdxOrValue()
                if k == '__idx':
                    idx_or_value.idx = v
                elif k == '__value':
                    idx_or_value.value.CopyFrom(ndarray_to_proto(v))
                else:
                    raise ValueError(f"Unknown key {k}")
                proto_change.field[k].CopyFrom(idx_or_value)
            proto_changes.changes.append(proto_change)
        return proto_changes
    elif isinstance(changes, dict):
        proto_state_change = simulator_pb2.StateChange()
        for attr, child in changes.items():
            x = changes_to_proto(child)
            if isinstance(x, simulator_pb2.Changes):
                nested = simulator_pb2.StateChange()
                nested.changes.CopyFrom(x)
                proto_state_change.child[attr].CopyFrom(nested)
            else:
                proto_state_change.child[attr].CopyFrom(changes_to_proto(child))
        return proto_state_change
    else:
        proto_state_change_list = simulator_pb2.StateChangeList()
        for change in changes:
            proto_state_change_list.state_changes.append(changes_to_proto(change))
        return proto_state_change_list
    
def proto_to_changes(proto_changes):
    if isinstance(proto_changes, simulator_pb2.StateChangeList):
        changes = []
        for proto_state_change in proto_changes.state_changes:
            change = proto_to_changes(proto_state_change)
            changes.append(change)
        return changes
    elif isinstance(proto_changes, simulator_pb2.StateChange):
        if proto_changes.HasField('changes'):
            return proto_to_changes(proto_changes.changes)
        else:
            changes = {}
            for attr, child in proto_changes.child.items():
                changes[attr] = proto_to_changes(child)
            return changes
    elif isinstance(proto_changes, simulator_pb2.Changes):
        changes = []
        for proto_change in proto_changes.changes:
            changes.append(proto_to_changes(proto_change))
        return changes
    elif isinstance(proto_changes, simulator_pb2.Change):
        change = {}
        for k, v in proto_changes.field.items():
            idx_or_value = v
            if k == '__idx':
                change['__idx'] = idx_or_value.idx
            elif k == '__value':
                change['__value'] = proto_to_ndarray(idx_or_value.value)
            else:
                raise ValueError(f"Unknown key {k}")
        return change



    if proto_changes.HasField('changes'):
        for proto_change in proto_changes.changes:
            change = {}
            for k in ['idx', 'value']:
                idx_or_value = getattr(proto_change, k)
                if k == 'idx':
                    change[k] = idx_or_value.idx
                elif k == 'value':
                    change[k] = proto_to_ndarray(idx_or_value.value)
                else:
                    raise ValueError(f"Unknown key {k}")
            changes.append(change)
    else:
        for proto_state_change in proto_changes.state_changes:
            change = {}
            for attr, child in proto_state_change.child.items():
                change[attr] = proto_to_changes(child)
            changes.append(change)
    return changes


def proto_to_state(state, dataclass_type):
    """Convert a protobuf state to a State object.

    :param state: simulation state in protobuf format
    :return: State object
    """

    if dataclass_type in [np.ndarray, jnp.ndarray]:
        return proto_to_ndarray(state.array_data)
    elif 'center' in state.nested_fields and 'orientation' in state.nested_fields:
        return RigidBody(
            center=proto_to_ndarray(state.nested_fields['center'].array_data).astype(float),
            orientation=proto_to_ndarray(state.nested_fields['orientation'].array_data).astype(float),
        )
    else:
        kwargs = {}
        for field in fields(dataclass_type):
            kwargs[field.name] = proto_to_state(state.nested_fields[field.name], field.type)

        return dataclass_type(**kwargs)



def state_to_proto(state):
    """Convert a State object to a protobuf state.

    :param state: simulation state
    :return: protobuf state
    """

    message = simulator_pb2.Dataclass()

    if isinstance(state, (np.ndarray, jnp.ndarray)):
        message.array_data.CopyFrom(ndarray_to_proto(state))
    else:
        for field in fields(state):
            value = getattr(state, field.name)
            message.nested_fields[field.name].CopyFrom(state_to_proto(value))
    return message

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
        previous_force=RigidBody(
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


# def state_to_proto(state):
#     """Convert a State object to a protobuf state.

#     :param state: simulation state
#     :return: protobuf state
#     """
#     return simulator_pb2.State(
#         simulator_state=simulator_state_to_proto(state.simulator_state),
#         entity_state=nve_state_to_proto(state.entity_state),
#         agent_state=agent_state_to_proto(state.agent_state),
#         object_state=object_state_to_proto(state.object_state),
#     )



    # return simulator_pb2.State(
    #     simulator_state=simulator_state_to_proto(state.simulator_state),
    #     entity_state=nve_state_to_proto(state.entity_state),
    #     agent_state=agent_state_to_proto(state.agent_state),
    #     object_state=object_state_to_proto(state.object_state),
    # )

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
            center=ndarray_to_proto(entity_state.position_center),
            orientation=ndarray_to_proto(entity_state.position_orientation),
        ),
        momentum=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.momentum_center),
            orientation=ndarray_to_proto(entity_state.momentum_orientation),
        ),
        force=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.force_center),
            orientation=ndarray_to_proto(entity_state.force_orientation),
        ),
        mass=simulator_pb2.RigidBody(
            center=ndarray_to_proto(entity_state.mass_center),
            orientation=ndarray_to_proto(entity_state.mass_orientation),
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
