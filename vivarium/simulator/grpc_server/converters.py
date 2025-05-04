from dataclasses import is_dataclass
import numpy as np

import jax.numpy as jnp

from jax_md.rigid_body import RigidBody
from jax_md.dataclasses import fields

import simulator_pb2

from vivarium.simulator.grpc_server.numproto.numproto import (
    proto_to_ndarray,
    ndarray_to_proto,
)


def idx_to_proto(idx):
    proto_index = simulator_pb2.Index()
    if isinstance(idx, (int, np.int32)):
        proto_index.idx.CopyFrom(simulator_pb2.Idx(int_idx=idx))
    elif isinstance(idx, slice):
        proto_index.slice_idx.start = idx.start
        proto_index.slice_idx.stop = idx.stop
        proto_index.slice_idx.step = idx.step
    else:
        raise ValueError(f"Unknown index type {type(idx)}")
    return proto_index


def indexes_to_proto(indexes):
    if isinstance(indexes, (int, np.int32, slice)):
        return idx_to_proto(indexes)
    #     index = simulator_pb2.Index()
    #     index.idx.CopyFrom(simulator_pb2.Idx(int_idx=indexes))
    #     return index
    # # proto_indexes = simulator_pb2.Indexes()
    # if isinstance(indexes, slice):
    #     proto_indexes.idx.append(idx_to_proto(indexes))
    elif isinstance(indexes, tuple):
        proto_indexes = simulator_pb2.Indexes()
        for idx in indexes:
            proto_indexes.idx.append(simulator_pb2.Idx(int_idx=idx))
        return simulator_pb2.Index(indexes=proto_indexes)
    elif indexes is None:
        index = simulator_pb2.Index()
        index.idx.CopyFrom(simulator_pb2.Idx(is_none=True))
        return index
        # proto_indexes.idx.append(simulator_pb2.Idx(is_none=True))
    else:
        raise ValueError(f"Unknown index type {type(indexes)}")
    # return proto_indexes


def changes_to_proto(changes):
    if isinstance(changes, list) and '__idx' in changes[0]:
        proto_changes = simulator_pb2.Changes()
        for change in changes:
            proto_change = simulator_pb2.Change()
            # idx = indexes_to_proto(change['__idx'])
            proto_change.idx.CopyFrom(indexes_to_proto(change['__idx']))
            if isinstance(change['__value'], (np.ndarray, jnp.ndarray)):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(ndarray=ndarray_to_proto(change['__value']))
                )
            elif isinstance(change['__value'], (float, np.float32, jnp.float32)):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(float_value=change['__value'])
                )
            elif isinstance(change['__value'], bool):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(bool_value=change['__value'])
                )
            elif isinstance(change['__value'], int):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(int_value=change['__value'])
                )
            elif isinstance(change['__value'], str):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(str_value=change['__value'])
                )
            elif isinstance(change['__value'], list):
                proto_change.value.CopyFrom(
                    simulator_pb2.Value(list_value=simulator_pb2.List(list=change['__value']))
                )
            else:
                raise ValueError(f"Unknown value type {type(change['__value'])}")
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


def proto_to_idx(proto_idx):
    if proto_idx.HasField('int_idx'):
        return proto_idx.int_idx
    elif proto_idx.HasField('slice_idx'):
        return slice(proto_idx.slice_idx.start, proto_idx.slice_idx.stop, proto_idx.slice_idx.step)
    elif proto_idx.HasField('is_none'):
        return None
    else:
        raise ValueError(f"Unknown index type {proto_idx}")


def proto_to_indexes(proto_indexes):
    indexes = []
    if proto_indexes.HasField('indexes'):
        for proto_idx in proto_indexes.indexes.idx:
            idx = proto_to_idx(proto_idx)
            indexes.append(idx)
        return tuple(indexes)
    if proto_indexes.HasField('idx'):
        return proto_to_idx(proto_indexes.idx)
    raise ValueError(f"Unknown index type {proto_indexes}")


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
        if proto_changes.value.HasField('ndarray'):
            value = proto_to_ndarray(proto_changes.value.ndarray)
        elif proto_changes.value.HasField('float_value'):
            value = proto_changes.value.float_value
        elif proto_changes.value.HasField('int_value'):
            value = proto_changes.value.int_value
        elif proto_changes.value.HasField('bool_value'):
            value = proto_changes.value.bool_value
        elif proto_changes.value.HasField('str_value'):
            value = proto_changes.value.str_value
        elif proto_changes.value.HasField('list_value'):
            value = list(proto_changes.value.list_value.list)
        change = {
            '__idx': proto_to_indexes(proto_changes.idx),
            '__value': value
        }
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

    if is_dataclass(dataclass_type):
        kwargs = {}
        for field in fields(dataclass_type):
            kwargs[field.name] = proto_to_state(state.nested_fields[field.name], field.type)
        return dataclass_type(**kwargs)
    elif state.value.HasField('int_value'):
        return state.value.int_value
    elif state.value.HasField('float_value'):
        return state.value.float_value
    elif state.value.HasField('bool_value'):
        return state.value.bool_value
    elif state.value.HasField('str_value'):
        return state.value.str_value
    elif state.value.HasField('list_value'):
        return state.value.list_value
    elif state.value.HasField('ndarray'):
        return proto_to_ndarray(state.value.ndarray)
    elif 'center' in state.nested_fields and 'orientation' in state.nested_fields:
        return RigidBody(
            center=proto_to_ndarray(state.nested_fields['center'].array_data).astype(float),
            orientation=proto_to_ndarray(state.nested_fields['orientation'].array_data).astype(float),
        )


def state_to_proto(state):
    """Convert a State object to a protobuf state.

    :param state: simulation state
    :return: protobuf state
    """

    message = simulator_pb2.Dataclass()

    if isinstance(state, (np.ndarray, jnp.ndarray)):
        message.value.CopyFrom(
                    simulator_pb2.Value(ndarray=ndarray_to_proto(state))
                )
    elif isinstance(state, bool):
        message.value.CopyFrom(
                    simulator_pb2.Value(bool_value=state)
                )
    elif isinstance(state, list):
        message.value.CopyFrom(
                    simulator_pb2.Value(list_value=state)
                )
    elif isinstance(state, int):
        message.value.CopyFrom(
                    simulator_pb2.Value(int_value=state)
                )
    elif isinstance(state, float):
        message.value.CopyFrom(
                    simulator_pb2.Value(float_value=state)
                )
    elif isinstance(state, str):
        message.value.CopyFrom(
                    simulator_pb2.Value(str_value=state)
                )
    else:
        for field in fields(state):
            value = getattr(state, field.name)
            message.nested_fields[field.name].CopyFrom(state_to_proto(value))
    return message
