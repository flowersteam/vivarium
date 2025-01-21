from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class AgentIdx(_message.Message):
    __slots__ = ("idx",)
    IDX_FIELD_NUMBER: _ClassVar[int]
    idx: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, idx: _Optional[_Iterable[int]] = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ("ndarray",)
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    ndarray: bytes
    def __init__(self, ndarray: _Optional[bytes] = ...) -> None: ...

class RigidBody(_message.Message):
    __slots__ = ("center", "orientation")
    CENTER_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    center: NDArray
    orientation: NDArray
    def __init__(
        self,
        center: _Optional[_Union[NDArray, _Mapping]] = ...,
        orientation: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class SimulatorState(_message.Message):
    __slots__ = (
        "idx",
        "box_size",
        "max_agents",
        "max_objects",
        "num_steps_lax",
        "dt",
        "freq",
        "neighbor_radius",
        "to_jit",
        "use_fori_loop",
        "collision_eps",
        "collision_alpha",
        "time",
    )
    IDX_FIELD_NUMBER: _ClassVar[int]
    BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_AGENTS_FIELD_NUMBER: _ClassVar[int]
    MAX_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    FREQ_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    COLLISION_EPS_FIELD_NUMBER: _ClassVar[int]
    COLLISION_ALPHA_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    idx: NDArray
    box_size: NDArray
    max_agents: NDArray
    max_objects: NDArray
    num_steps_lax: NDArray
    dt: NDArray
    freq: NDArray
    neighbor_radius: NDArray
    to_jit: NDArray
    use_fori_loop: NDArray
    collision_eps: NDArray
    collision_alpha: NDArray
    time: NDArray
    def __init__(
        self,
        idx: _Optional[_Union[NDArray, _Mapping]] = ...,
        box_size: _Optional[_Union[NDArray, _Mapping]] = ...,
        max_agents: _Optional[_Union[NDArray, _Mapping]] = ...,
        max_objects: _Optional[_Union[NDArray, _Mapping]] = ...,
        num_steps_lax: _Optional[_Union[NDArray, _Mapping]] = ...,
        dt: _Optional[_Union[NDArray, _Mapping]] = ...,
        freq: _Optional[_Union[NDArray, _Mapping]] = ...,
        neighbor_radius: _Optional[_Union[NDArray, _Mapping]] = ...,
        to_jit: _Optional[_Union[NDArray, _Mapping]] = ...,
        use_fori_loop: _Optional[_Union[NDArray, _Mapping]] = ...,
        collision_eps: _Optional[_Union[NDArray, _Mapping]] = ...,
        collision_alpha: _Optional[_Union[NDArray, _Mapping]] = ...,
        time: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class EntityState(_message.Message):
    __slots__ = (
        "position",
        "momentum",
        "force",
        "mass",
        "diameter",
        "entity_type",
        "entity_idx",
        "friction",
        "exists",
        "ent_subtype",
    )
    POSITION_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    DIAMETER_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_IDX_FIELD_NUMBER: _ClassVar[int]
    FRICTION_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    ENT_SUBTYPE_FIELD_NUMBER: _ClassVar[int]
    position: RigidBody
    momentum: RigidBody
    force: RigidBody
    mass: RigidBody
    diameter: NDArray
    entity_type: NDArray
    entity_idx: NDArray
    friction: NDArray
    exists: NDArray
    ent_subtype: NDArray
    def __init__(
        self,
        position: _Optional[_Union[RigidBody, _Mapping]] = ...,
        momentum: _Optional[_Union[RigidBody, _Mapping]] = ...,
        force: _Optional[_Union[RigidBody, _Mapping]] = ...,
        mass: _Optional[_Union[RigidBody, _Mapping]] = ...,
        diameter: _Optional[_Union[NDArray, _Mapping]] = ...,
        entity_type: _Optional[_Union[NDArray, _Mapping]] = ...,
        entity_idx: _Optional[_Union[NDArray, _Mapping]] = ...,
        friction: _Optional[_Union[NDArray, _Mapping]] = ...,
        exists: _Optional[_Union[NDArray, _Mapping]] = ...,
        ent_subtype: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class AgentState(_message.Message):
    __slots__ = (
        "ent_idx",
        "prox",
        "motor",
        "behavior",
        "wheel_diameter",
        "speed_mul",
        "max_speed",
        "theta_mul",
        "proxs_dist_max",
        "proxs_cos_min",
        "color",
        "proximity_map_dist",
        "proximity_map_theta",
        "params",
        "sensed",
        "prox_sensed_ent_type",
        "prox_sensed_ent_idx",
    )
    ENT_IDX_FIELD_NUMBER: _ClassVar[int]
    PROX_FIELD_NUMBER: _ClassVar[int]
    MOTOR_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    PROXIMITY_MAP_DIST_FIELD_NUMBER: _ClassVar[int]
    PROXIMITY_MAP_THETA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    SENSED_FIELD_NUMBER: _ClassVar[int]
    PROX_SENSED_ENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROX_SENSED_ENT_IDX_FIELD_NUMBER: _ClassVar[int]
    ent_idx: NDArray
    prox: NDArray
    motor: NDArray
    behavior: NDArray
    wheel_diameter: NDArray
    speed_mul: NDArray
    max_speed: NDArray
    theta_mul: NDArray
    proxs_dist_max: NDArray
    proxs_cos_min: NDArray
    color: NDArray
    proximity_map_dist: NDArray
    proximity_map_theta: NDArray
    params: NDArray
    sensed: NDArray
    prox_sensed_ent_type: NDArray
    prox_sensed_ent_idx: NDArray
    def __init__(
        self,
        ent_idx: _Optional[_Union[NDArray, _Mapping]] = ...,
        prox: _Optional[_Union[NDArray, _Mapping]] = ...,
        motor: _Optional[_Union[NDArray, _Mapping]] = ...,
        behavior: _Optional[_Union[NDArray, _Mapping]] = ...,
        wheel_diameter: _Optional[_Union[NDArray, _Mapping]] = ...,
        speed_mul: _Optional[_Union[NDArray, _Mapping]] = ...,
        max_speed: _Optional[_Union[NDArray, _Mapping]] = ...,
        theta_mul: _Optional[_Union[NDArray, _Mapping]] = ...,
        proxs_dist_max: _Optional[_Union[NDArray, _Mapping]] = ...,
        proxs_cos_min: _Optional[_Union[NDArray, _Mapping]] = ...,
        color: _Optional[_Union[NDArray, _Mapping]] = ...,
        proximity_map_dist: _Optional[_Union[NDArray, _Mapping]] = ...,
        proximity_map_theta: _Optional[_Union[NDArray, _Mapping]] = ...,
        params: _Optional[_Union[NDArray, _Mapping]] = ...,
        sensed: _Optional[_Union[NDArray, _Mapping]] = ...,
        prox_sensed_ent_type: _Optional[_Union[NDArray, _Mapping]] = ...,
        prox_sensed_ent_idx: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class ObjectState(_message.Message):
    __slots__ = ("ent_idx", "custom_field", "color")
    ENT_IDX_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    ent_idx: NDArray
    custom_field: NDArray
    color: NDArray
    def __init__(
        self,
        ent_idx: _Optional[_Union[NDArray, _Mapping]] = ...,
        custom_field: _Optional[_Union[NDArray, _Mapping]] = ...,
        color: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class State(_message.Message):
    __slots__ = ("simulator_state", "entity_state", "agent_state", "object_state")
    SIMULATOR_STATE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_STATE_FIELD_NUMBER: _ClassVar[int]
    AGENT_STATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STATE_FIELD_NUMBER: _ClassVar[int]
    simulator_state: SimulatorState
    entity_state: EntityState
    agent_state: AgentState
    object_state: ObjectState
    def __init__(
        self,
        simulator_state: _Optional[_Union[SimulatorState, _Mapping]] = ...,
        entity_state: _Optional[_Union[EntityState, _Mapping]] = ...,
        agent_state: _Optional[_Union[AgentState, _Mapping]] = ...,
        object_state: _Optional[_Union[ObjectState, _Mapping]] = ...,
    ) -> None: ...

class StateChange(_message.Message):
    __slots__ = ("ent_idx", "col_idx", "nested_field", "value")
    ENT_IDX_FIELD_NUMBER: _ClassVar[int]
    COL_IDX_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ent_idx: _containers.RepeatedScalarFieldContainer[int]
    col_idx: _containers.RepeatedScalarFieldContainer[int]
    nested_field: _containers.RepeatedScalarFieldContainer[str]
    value: NDArray
    def __init__(
        self,
        ent_idx: _Optional[_Iterable[int]] = ...,
        col_idx: _Optional[_Iterable[int]] = ...,
        nested_field: _Optional[_Iterable[str]] = ...,
        value: _Optional[_Union[NDArray, _Mapping]] = ...,
    ) -> None: ...

class AddAgentInput(_message.Message):
    __slots__ = ("max_agents", "serialized_config")
    MAX_AGENTS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_CONFIG_FIELD_NUMBER: _ClassVar[int]
    max_agents: int
    serialized_config: str
    def __init__(
        self, max_agents: _Optional[int] = ..., serialized_config: _Optional[str] = ...
    ) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ("is_started",)
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...

class SubtypesLabels(_message.Message):
    __slots__ = ("data",)

    class DataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: str
        def __init__(
            self, key: _Optional[int] = ..., value: _Optional[str] = ...
        ) -> None: ...

    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.ScalarMap[int, str]
    def __init__(self, data: _Optional[_Mapping[int, str]] = ...) -> None: ...

class Scene(_message.Message):
    __slots__ = ("scene_name",)
    SCENE_NAME_FIELD_NUMBER: _ClassVar[int]
    scene_name: str
    def __init__(self, scene_name: _Optional[str] = ...) -> None: ...
