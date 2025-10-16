from dataclasses import is_dataclass
import numpy as np

import jax.numpy as jnp

from jax_md.rigid_body import RigidBody
from jax_md.dataclasses import fields

from vivarium.controllers.dataclass_wrapper import create_dataclass_from_dict

import simulator_pb2

from vivarium.simulator.grpc_server.numproto.numproto import (
    proto_to_ndarray,
    ndarray_to_proto,
)

def slice_args_to_proto(arg):
    if arg is None:
        return simulator_pb2.SliceArg(is_none=True)
    else:
        return simulator_pb2.SliceArg(int_arg=arg)

def idx_to_proto(idx):
    if isinstance(idx, (int, np.int32)):
        return simulator_pb2.Idx(int_idx=idx)
    elif isinstance(idx, (tuple, list)):
        proto_indexes = simulator_pb2.IndexList()
        for i in idx:
            proto_indexes.index_list.append(idx_to_proto(i))
        return simulator_pb2.Idx(index_list=proto_indexes)    
    elif isinstance(idx, slice):
        return simulator_pb2.Idx(slice_idx=simulator_pb2.Slice(
            start=slice_args_to_proto(idx.start),
            stop=slice_args_to_proto(idx.stop),
            step=slice_args_to_proto(idx.step)
        ))
    elif idx is None:
        return simulator_pb2.Idx(is_none=True)
    else:
        raise ValueError(f"Unknown index type {type(idx)}")


def changes_to_proto(changes):
    if isinstance(changes, list) and len(changes) > 0 and '__idx' in changes[0]:
        proto_changes = simulator_pb2.Changes()
        for change in changes:
            proto_change = simulator_pb2.Change()
            proto_change.idx.CopyFrom(idx_to_proto(change['__idx']))
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
                if isinstance(change['__value'][0], bool):
                    value = simulator_pb2.Value(
                        list_bool_value=simulator_pb2.ListBool(list=change['__value'])
                    )                
                elif isinstance(change['__value'][0], (float, int)):
                    value = simulator_pb2.Value(
                        list_float_value=simulator_pb2.ListFloat(list=change['__value'])
                    )
                elif isinstance(change['__value'][0], str):
                    value = simulator_pb2.Value(
                        list_string_value=simulator_pb2.ListString(list=change['__value'])
                    )
                else:
                    raise ValueError(f"List items of type {type(change['__value'][0])} not supported yet.")
                proto_change.value.CopyFrom(value)
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


def proto_to_slice_arg(proto_arg):
    if proto_arg.HasField('int_arg'):
        return proto_arg.int_arg
    elif proto_arg.HasField('is_none'):
        return None
    else:
        raise ValueError(f"Unknown slice argument type {proto_arg}")

def proto_to_idx(proto_idx):
    if proto_idx.HasField('int_idx'):
        return proto_idx.int_idx
    elif proto_idx.HasField('index_list'):
        indexes = []
        for idx in proto_idx.index_list.index_list:
            indexes.append(proto_to_idx(idx))
        return tuple(indexes)
    elif proto_idx.HasField('slice_idx'):
        return slice(proto_to_slice_arg(proto_idx.slice_idx.start), 
                     proto_to_slice_arg(proto_idx.slice_idx.stop), 
                     proto_to_slice_arg(proto_idx.slice_idx.step)
                     )
    elif proto_idx.HasField('is_none'):
        return None
    else:
        raise ValueError(f"Unknown index type {proto_idx}")


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
        elif proto_changes.value.HasField('list_float_value'):
            value = list(proto_changes.value.list_float_value.list)
        elif proto_changes.value.HasField('list_string_value'):
            value = list(proto_changes.value.list_string_value.list)
        elif proto_changes.value.HasField('list_bool_value'):
            value = list(proto_changes.value.list_bool_value.list)
        change = {
            '__idx': proto_to_idx(proto_changes.idx),
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


def proto_to_dataclass(dataclass, dataclass_type=None):
    """Convert a protobuf to a dataclass instance.

    :param dataclass: simulator_pb2.Dataclass message
    :param dataclass_type: The type of the dataclass to convert to. 
    If None, a generic dataclass is created. In this case class methods won't be accessible after deserialization.
    :return: python dataclass instance
    """

    if len(dataclass.nested_fields) > 0:
        kwargs = {}
        if dataclass_type is not None:
            for field in fields(dataclass_type):
                kwargs[field.name] = proto_to_dataclass(dataclass.nested_fields[field.name], field.type)
            return dataclass_type(**kwargs)
        else: # In this case class methods won't be accessible
            for field, value in dataclass.nested_fields.items():
                kwargs[field] = proto_to_dataclass(value) #, None)
            return create_dataclass_from_dict('FromProto', kwargs)
    elif dataclass.value.HasField('int_value'):
        return dataclass.value.int_value
    elif dataclass.value.HasField('float_value'):
        return dataclass.value.float_value
    elif dataclass.value.HasField('bool_value'):
        return dataclass.value.bool_value
    elif dataclass.value.HasField('str_value'):
        return dataclass.value.str_value
    elif dataclass.value.HasField('list_float_value'):
        return dataclass.value.list_float_value.list
    elif dataclass.value.HasField('list_string_value'):
        return dataclass.value.list_string_value.list
    elif dataclass.value.HasField('list_bool_value'):
        return dataclass.value.list_bool_value.list
    elif dataclass.value.HasField('list_behaviors_value'):
        list_behaviors = []
        for behaviors in dataclass.value.list_behaviors_value.list:
            list_behavior = []
            for behavior in behaviors.behaviors:
                behavior_dict = {}
                for label, sensed in behavior.behavior_to_sensed.items():
                    behavior_dict[label] = list(sensed.list)
                list_behavior.append(behavior_dict)
            list_behaviors.append(list_behavior)
        return list_behaviors
    elif dataclass.value.HasField('ndarray'):
        return proto_to_ndarray(dataclass.value.ndarray)
    elif 'center' in dataclass.nested_fields and 'orientation' in dataclass.nested_fields:
        return RigidBody(
            center=proto_to_ndarray(dataclass.nested_fields['center'].array_data).astype(float),
            orientation=proto_to_ndarray(dataclass.nested_fields['orientation'].array_data).astype(float),
        )


def dataclass_to_proto(dataclass):
    """Convert a dataclass object to a protobuf message.

    :param dataclass: python dataclass instance
    :return: simulator_pb2.Dataclass message
    """

    message = simulator_pb2.Dataclass()

    if isinstance(dataclass, (np.ndarray, jnp.ndarray)):
        message.value.CopyFrom(
                    simulator_pb2.Value(ndarray=ndarray_to_proto(dataclass))
                )
    elif isinstance(dataclass, bool):
        message.value.CopyFrom(
                    simulator_pb2.Value(bool_value=dataclass)
                )
    elif isinstance(dataclass, list):
        if isinstance(dataclass[0], bool):
            value = simulator_pb2.Value(list_bool_value=simulator_pb2.ListBool(list=dataclass))
        elif isinstance(dataclass[0], (float, int)):
            value = simulator_pb2.Value(list_float_value=simulator_pb2.ListFloat(list=dataclass))
        elif isinstance(dataclass[0], str):
            value = simulator_pb2.Value(list_string_value=simulator_pb2.ListString(list=dataclass))
        elif isinstance(dataclass[0], list) and isinstance(dataclass[0][0], dict):
            value = simulator_pb2.Value(list_behaviors_value=simulator_pb2.ListBehaviors())
            for item in dataclass:
                behaviors = simulator_pb2.Behaviors()
                for behavior in item:
                    b = simulator_pb2.Behavior()
                    for label, sensed in behavior.items():
                        b.behavior_to_sensed[label].CopyFrom(simulator_pb2.ListString(list=sensed))
                    behaviors.behaviors.append(b)
                value.list_behaviors_value.list.append(behaviors)
        else:
            raise ValueError(f"List items of type {type(dataclass[0])} not supported yet.")
        message.value.CopyFrom(value)
    elif isinstance(dataclass, int):
        message.value.CopyFrom(
                    simulator_pb2.Value(int_value=dataclass)
                )
    elif isinstance(dataclass, float):
        message.value.CopyFrom(
                    simulator_pb2.Value(float_value=dataclass)
                )
    elif isinstance(dataclass, str):
        message.value.CopyFrom(
                    simulator_pb2.Value(str_value=dataclass)
                )
    else:
        for field in fields(dataclass):
            value = getattr(dataclass, field.name)
            message.nested_fields[field.name].CopyFrom(dataclass_to_proto(value))
    return message
