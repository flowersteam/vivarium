"""
This module provides functions and data structures to convert between different configurations
and states in a simulation environment. It handles the conversion of agent, object, and simulator
configurations to their respective state representations and vice versa. The module also includes
utility functions to handle events and state changes within the simulation.
"""

import typing
import logging
import dataclasses
from collections import namedtuple, defaultdict

import jax_md
import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors

from vivarium.controllers.config import (
    AgentConfig,
    ObjectConfig,
    SimulatorConfig,
    stype_to_config,
    config_to_stype,
)
from vivarium.simulator.simulator_states import (
    SimulatorState,
    AgentState,
    ObjectState,
    StateType,
)

lg = logging.getLogger(__name__)

if logging.root.handlers:
    lg.setLevel(logging.root.level)


@dataclasses.dataclass
class StateFieldInfo:
    nested_field: typing.Tuple
    column_idx: np.array
    state_to_config: typing.Callable
    config_to_state: typing.Callable


EntityTuple = namedtuple("EntityTuple", ["idx", "col", "val"])
ValueTuple = namedtuple(
    "ValueData", ["ent_idx", "col_idx", "row_map", "col_map", "val"]
)
StateChangeTuple = namedtuple(
    "StateChange", ["nested_field", "ent_idx", "column_idx", "value"]
)


def events_to_nve_data(events, state):
    """Convert events to a dictionary of nested field to entity tuples.
    NVE corresponds to this jax-md naming convention: https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nve

    :param events: list of events to convert
    :param state: current state of the simulation
    :return: dictionary of nested field to entity tuples
    """
    nve_data = defaultdict(list)
    for e in events:
        config = e.obj
        param = e.name
        etype = config_to_stype[config.param.cls]
        state_field_info = configs_to_state_dict[etype][param]
        nested_field = state_field_info.nested_field
        idx = config.idx
        val = state_field_info.config_to_state(e.new)

        if state_field_info.column_idx is None:
            nve_data[nested_field].append(EntityTuple(idx, None, val))
        elif isinstance(state_field_info.column_idx, int):
            nve_data[nested_field].append(
                EntityTuple(idx, state_field_info.column_idx, val)
            )
        elif isinstance(state_field_info.column_idx, slice):
            nve_data[nested_field].append(
                EntityTuple(idx, state_field_info.column_idx, val)
            )
        else:
            for c, v in zip(state_field_info.column_idx, val):
                nve_data[nested_field].append(EntityTuple(idx, c, v))

    return nve_data


def nve_data_to_state_changes(nve_data, state):
    """Convert nve data to a list of state changes

    :param nve_data: nve data
    :param state: current state of the simulation
    :return: list of state changes
    """
    value_data = dict()
    for nf, nve_tuples in nve_data.items():
        ent_idx = sorted(list(set([int(t.idx) for t in nve_tuples])))
        row_map = {idx: i for i, idx in enumerate(ent_idx)}
        # Check if the colum is an int, a slice or None
        if nve_tuples[0].col is None:
            val = np.array(state.field(nf)[np.array(ent_idx)])
            col_map = None
            col_idx = None
        elif isinstance(nve_tuples[0].col, slice):
            val = np.array(state.field(nf)[np.array(ent_idx)])
            col_map = nve_tuples[0].col
            col_idx = None
        else:
            col_idx = sorted(list(set([t.col for t in nve_tuples])))
            col_map = {idx: i for i, idx in enumerate(col_idx)}
            val = np.array(
                state.field(nf)[np.ix_(state.row_idx(nf[0], ent_idx), col_idx)]
            )
        value_data[nf] = ValueTuple(ent_idx, col_idx, row_map, col_map, val)

    state_changes = []
    for nf, value_tuple in value_data.items():
        for nve_tuple in nve_data[nf]:
            row = value_tuple.row_map[nve_tuple.idx]
            # Check if the colum is an int, a slice or None
            if nve_tuple.col is None:
                value_tuple.val[row] = nve_tuple.val
            elif isinstance(nve_tuple.col, slice):
                value_tuple.val[row] = nve_tuple.val
            else:
                col = value_tuple.col_map[nve_tuple.col]
                value_tuple.val[row, col] = nve_tuple.val
        state_changes.append(
            StateChangeTuple(
                nf, value_data[nf].ent_idx, value_data[nf].col_idx, value_tuple.val
            )
        )

    return state_changes


def events_to_state_changes(events, state):
    """Convert events to a list of state changes

    :param events: list of events to convert
    :param state: current state of the simulation
    :return: list of state changes
    """
    nve_data = events_to_nve_data(events, state)
    return nve_data_to_state_changes(nve_data, state)


def rec_set_dataclass(var, nested_field, row_idx, column_idx, value):
    """Set a value in a nested dataclass

    :param var: variable
    :param nested_field: nested field to set
    :param row_idx: row index of the value to set
    :param column_idx: column index of the value to set
    :param value: value to set
    :return: dictionary with the value set
    """
    assert len(nested_field) > 0
    if isinstance(column_idx, int):
        column_idx = np.array([column_idx])

    if len(nested_field) == 1:
        field = nested_field[0]
        if isinstance(column_idx, slice):
            column_idx = np.arange(
                column_idx.start if column_idx.start is not None else 0,
                (
                    column_idx.stop
                    if column_idx.stop is not None
                    else getattr(var, field).shape[1]
                ),
                column_idx.step if column_idx.step is not None else 1,
                dtype=int,
            )
        if column_idx is None or len(column_idx) == 0:
            d = {field: getattr(var, field).at[row_idx].set(value.reshape(-1))}
        else:
            d = {
                field: getattr(var, field)
                .at[jnp.ix_(row_idx, column_idx)]
                .set(value.reshape(len(row_idx), len(column_idx)))
            }
        return d
    else:
        next_var = getattr(var, nested_field[0])
        d = rec_set_dataclass(next_var, nested_field[1:], row_idx, column_idx, value)
        return {nested_field[0]: next_var.set(**d)}


def set_configs_from_state(state, config_dict=None):
    """Set the configuration dictionary from the state

    :param state: current state of the simulation
    :param config_dict: current dict of configurations, defaults to None
    :return: updated dict of configurations
    """
    if config_dict is None:
        config_dict = {stype: [] for stype in list(StateType)}
        for idx, stype_int in enumerate(state.entity_state.entity_type):
            stype = StateType(stype_int)
            config_dict[stype].append(stype_to_config[stype](idx=idx))
        config_dict[StateType.SIMULATOR].append(SimulatorConfig())

    for stype in config_dict.keys():
        for param, state_field_info in configs_to_state_dict[stype].items():
            value = state
            for f in state_field_info.nested_field:
                value = getattr(value, f)
            for config in config_dict[stype]:
                t = type(getattr(config, param))
                row_idx = state.row_idx(state_field_info.nested_field[0], config.idx)
                if state_field_info.column_idx is None:
                    value_to_set = value[row_idx]
                else:
                    value_to_set = value[row_idx, state_field_info.column_idx]
                # lg.debug(f"Setting {param} to {value_to_set} for {stype} {config.idx}")
                value_to_set = state_field_info.state_to_config(value_to_set, t)
                config.param.update(**{param: value_to_set})
    return config_dict


# Define conversions between entities configs and state dictionary
identity_s_to_c = lambda x, typ: typ(x)
identity_c_to_s = lambda x: x
color_s_to_c = lambda x, typ: mcolors.to_hex(np.array(x))
color_c_to_s = lambda x: mcolors.to_rgb(x)
mass_center_s_to_c = lambda x, typ: typ(x)
mass_center_c_to_s = lambda x: [x]
int_c_to_s = lambda x: int(x)
array_map_s_to_c = lambda x, typ: x


# Define config fields for agents
agent_config_fields = AgentConfig.param.objects().keys()
agent_state_fields = [f.name for f in jax_md.dataclasses.fields(AgentState)]
agent_common_fields = [f for f in agent_config_fields if f in agent_state_fields]
# Define config fields for objects
object_config_fields = ObjectConfig.param.objects().keys()
object_state_fields = [f.name for f in jax_md.dataclasses.fields(ObjectState)]
object_common_fields = [f for f in object_config_fields if f in object_state_fields]
# Define config fields for simulator
simulator_config_fields = SimulatorConfig.param.objects().keys()
simulator_state_fields = [f.name for f in jax_md.dataclasses.fields(SimulatorState)]
simulator_common_fields = [
    f for f in simulator_config_fields if f in simulator_state_fields
]


# Define conversions between agents configs and state dictionary
agent_configs_to_state_dict = {
    "x_position": StateFieldInfo(
        ("entity_state", "position", "center"), 0, identity_s_to_c, identity_c_to_s
    ),
    "y_position": StateFieldInfo(
        ("entity_state", "position", "center"), 1, identity_s_to_c, identity_c_to_s
    ),
    "orientation": StateFieldInfo(
        ("entity_state", "position", "orientation"),
        None,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "mass_center": StateFieldInfo(
        ("entity_state", "mass", "center"),
        np.array([0]),
        mass_center_s_to_c,
        mass_center_c_to_s,
    ),
    "mass_orientation": StateFieldInfo(
        ("entity_state", "mass", "orientation"), None, identity_s_to_c, identity_c_to_s
    ),
    "diameter": StateFieldInfo(
        ("entity_state", "diameter"), None, identity_s_to_c, identity_c_to_s
    ),
    "friction": StateFieldInfo(
        ("entity_state", "friction"), None, identity_s_to_c, identity_c_to_s
    ),
    "left_motor": StateFieldInfo(
        (
            "agent_state",
            "motor",
        ),
        0,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "right_motor": StateFieldInfo(
        (
            "agent_state",
            "motor",
        ),
        1,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "left_prox": StateFieldInfo(
        (
            "agent_state",
            "prox",
        ),
        0,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "right_prox": StateFieldInfo(
        (
            "agent_state",
            "prox",
        ),
        1,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "prox_sensed_ent_type": StateFieldInfo(
        (
            "agent_state",
            "prox_sensed_ent_type",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "prox_sensed_ent_idx": StateFieldInfo(
        (
            "agent_state",
            "prox_sensed_ent_idx",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "proximity_map_dist": StateFieldInfo(
        (
            "agent_state",
            "proximity_map_dist",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "proximity_map_theta": StateFieldInfo(
        (
            "agent_state",
            "proximity_map_theta",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "params": StateFieldInfo(
        (
            "agent_state",
            "params",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "sensed": StateFieldInfo(
        (
            "agent_state",
            "sensed",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "behavior": StateFieldInfo(
        (
            "agent_state",
            "behavior",
        ),
        slice(None),
        array_map_s_to_c,
        identity_c_to_s,
    ),
    "color": StateFieldInfo(
        (
            "agent_state",
            "color",
        ),
        np.arange(3),
        color_s_to_c,
        color_c_to_s,
    ),
    "idx": StateFieldInfo(
        (
            "agent_state",
            "ent_idx",
        ),
        None,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "exists": StateFieldInfo(
        ("entity_state", "exists"), None, identity_s_to_c, int_c_to_s
    ),
    "subtype": StateFieldInfo(
        ("entity_state", "ent_subtype"), None, identity_s_to_c, int_c_to_s
    ),
}

agent_configs_to_state_dict.update(
    {
        f: StateFieldInfo(
            (
                "agent_state",
                f,
            ),
            None,
            identity_s_to_c,
            identity_c_to_s,
        )
        for f in agent_common_fields
        if f not in agent_configs_to_state_dict
    }
)

# Define conversions between objects configs and state dictionary
object_configs_to_state_dict = {
    "x_position": StateFieldInfo(
        ("entity_state", "position", "center"), 0, identity_s_to_c, identity_c_to_s
    ),
    "y_position": StateFieldInfo(
        ("entity_state", "position", "center"), 1, identity_s_to_c, identity_c_to_s
    ),
    "orientation": StateFieldInfo(
        ("entity_state", "position", "orientation"),
        None,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "mass_center": StateFieldInfo(
        ("entity_state", "mass", "center"),
        np.array([0]),
        mass_center_s_to_c,
        mass_center_c_to_s,
    ),
    "mass_orientation": StateFieldInfo(
        ("entity_state", "mass", "orientation"), None, identity_s_to_c, identity_c_to_s
    ),
    "diameter": StateFieldInfo(
        ("entity_state", "diameter"), None, identity_s_to_c, identity_c_to_s
    ),
    "friction": StateFieldInfo(
        ("entity_state", "friction"), None, identity_s_to_c, identity_c_to_s
    ),
    "color": StateFieldInfo(
        (
            "object_state",
            "color",
        ),
        np.arange(3),
        color_s_to_c,
        color_c_to_s,
    ),
    "idx": StateFieldInfo(
        (
            "object_state",
            "ent_idx",
        ),
        None,
        identity_s_to_c,
        identity_c_to_s,
    ),
    "exists": StateFieldInfo(
        ("entity_state", "exists"), None, identity_s_to_c, int_c_to_s
    ),
    "subtype": StateFieldInfo(
        ("entity_state", "ent_subtype"), None, identity_s_to_c, int_c_to_s
    ),
}

object_configs_to_state_dict.update(
    {
        f: StateFieldInfo(
            (
                "object_state",
                f,
            ),
            None,
            identity_s_to_c,
            identity_c_to_s,
        )
        for f in object_common_fields
        if f not in object_configs_to_state_dict
    }
)

simulator_configs_to_state_dict = {}
simulator_configs_to_state_dict.update(
    {
        f: StateFieldInfo(
            (
                "simulator_state",
                f,
            ),
            None,
            identity_s_to_c,
            identity_c_to_s,
        )
        for f in simulator_common_fields
        if f not in simulator_configs_to_state_dict
    }
)

# Define conversions between objects configs and state dictionary
configs_to_state_dict = {
    StateType.AGENT: agent_configs_to_state_dict,
    StateType.OBJECT: object_configs_to_state_dict,
    StateType.SIMULATOR: simulator_configs_to_state_dict,
}


# Define state fields for each state type
state_fields_dict = {
    StateType.AGENT: agent_state_fields,
    StateType.OBJECT: object_state_fields,
    StateType.SIMULATOR: simulator_state_fields,
}
