from enum import Enum

import jax.numpy as jnp

from jax_md.dataclasses import dataclass
from jax_md import util, simulate, rigid_body


class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

    def to_state_type(self):
        return StateType(self.value)


class StateType(Enum):
    AGENT = 0
    OBJECT = 1
    SIMULATOR = 2

    def is_entity(self):
        return self != StateType.SIMULATOR

    def to_entity_type(self):
        assert self.is_entity()
        return EntityType(self.value)


# No need to define position, momentum, force, and mass (i.e already in jax_md.simulate.NVEState)
@dataclass
class EntityState(simulate.NVEState):
    entity_type: util.Array
    ent_subtype: util.Array
    entity_idx: util.Array  # idx in XState (e.g. AgentState)
    diameter: util.Array
    friction: util.Array
    exists: util.Array

    @property
    def velocity(self) -> util.Array:
        return self.momentum / self.mass


# TODO : ent idx and color already in Particle state in env side


@dataclass
class AgentState:
    ent_idx: util.Array  # idx in EntityState
    prox: util.Array
    prox_sensed_ent_type: util.Array
    prox_sensed_ent_idx: util.Array
    # prox_sensed_type: util.Array
    motor: util.Array
    proximity_map_dist: util.Array
    proximity_map_theta: util.Array
    behavior: util.Array
    params: util.Array
    sensed: util.Array
    wheel_diameter: util.Array
    speed_mul: util.Array
    max_speed: util.Array
    theta_mul: util.Array
    proxs_dist_max: util.Array
    proxs_cos_min: util.Array
    color: util.Array


@dataclass
class ObjectState:
    ent_idx: util.Array  # idx in EntityState
    color: util.Array


@dataclass
class SimulatorState:
    idx: util.Array
    time: util.Array
    # ent_sub_types: util.Array # TODO : Maybe add it later
    box_size: util.Array
    max_agents: util.Array
    max_objects: util.Array
    num_steps_lax: util.Array
    dt: util.Array
    freq: util.Array
    neighbor_radius: util.Array
    to_jit: util.Array
    use_fori_loop: util.Array
    collision_alpha: util.Array
    collision_eps: util.Array

    # DONE : Added time
    @staticmethod
    def get_type(attr):
        if attr in ["idx", "max_agents", "max_objects", "num_steps_lax"]:
            return int
        elif attr in [
            "time",
            "box_size",
            "dt",
            "freq",
            "neighbor_radius",
            "collision_alpha",
            "collision_eps",
        ]:
            return float
        elif attr in ["to_jit", "use_fori_loop"]:
            return bool
        else:
            raise ValueError(f"Unknown attribute {attr}")


@dataclass
class SimState:
    simulator_state: SimulatorState
    entity_state: EntityState
    agent_state: AgentState
    object_state: ObjectState

    def field(self, stype_or_nested_fields):
        if isinstance(stype_or_nested_fields, StateType) or isinstance(
            stype_or_nested_fields, Enum
        ):
            name = stype_or_nested_fields.name.lower()
            nested_fields = (f"{name}_state",)
        else:
            nested_fields = stype_or_nested_fields
        # TODO : what does this line do ?
        res = self
        for f in nested_fields:
            res = getattr(res, f)

        return res

    def ent_idx(self, etype, entity_idx):
        return self.field(etype).ent_idx[entity_idx]

    def e_idx(self, etype):
        return self.entity_state.entity_idx[
            self.entity_state.entity_type == etype.value
        ]

    def e_cond(self, etype):
        return self.entity_state.entity_type == etype.value

    def row_idx(self, field, ent_idx):
        return (
            ent_idx
            if field == "entity_state"
            else self.entity_state.entity_idx[jnp.array(ent_idx)]
        )

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.entity_state, name)
            if isinstance(value, rigid_body.RigidBody):
                return rigid_body.RigidBody(
                    center=value.center[self.e_cond(e_type)],
                    orientation=value.orientation[self.e_cond(e_type)],
                )
            else:
                return value[self.e_cond(e_type)]

        return wrapper
