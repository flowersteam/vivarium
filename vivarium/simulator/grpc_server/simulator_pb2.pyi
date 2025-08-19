from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Dataclass(_message.Message):
    __slots__ = ("value", "nested_fields")
    class NestedFieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Dataclass
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Dataclass, _Mapping]] = ...) -> None: ...
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    value: Value
    nested_fields: _containers.MessageMap[str, Dataclass]
    def __init__(self, value: _Optional[_Union[Value, _Mapping]] = ..., nested_fields: _Optional[_Mapping[str, Dataclass]] = ...) -> None: ...

class SliceArg(_message.Message):
    __slots__ = ("int_arg", "is_none")
    INT_ARG_FIELD_NUMBER: _ClassVar[int]
    IS_NONE_FIELD_NUMBER: _ClassVar[int]
    int_arg: int
    is_none: bool
    def __init__(self, int_arg: _Optional[int] = ..., is_none: bool = ...) -> None: ...

class Slice(_message.Message):
    __slots__ = ("start", "stop", "step")
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    start: SliceArg
    stop: SliceArg
    step: SliceArg
    def __init__(self, start: _Optional[_Union[SliceArg, _Mapping]] = ..., stop: _Optional[_Union[SliceArg, _Mapping]] = ..., step: _Optional[_Union[SliceArg, _Mapping]] = ...) -> None: ...

class IndexList(_message.Message):
    __slots__ = ("index_list",)
    INDEX_LIST_FIELD_NUMBER: _ClassVar[int]
    index_list: _containers.RepeatedCompositeFieldContainer[Idx]
    def __init__(self, index_list: _Optional[_Iterable[_Union[Idx, _Mapping]]] = ...) -> None: ...

class Idx(_message.Message):
    __slots__ = ("int_idx", "slice_idx", "index_list", "is_none")
    INT_IDX_FIELD_NUMBER: _ClassVar[int]
    SLICE_IDX_FIELD_NUMBER: _ClassVar[int]
    INDEX_LIST_FIELD_NUMBER: _ClassVar[int]
    IS_NONE_FIELD_NUMBER: _ClassVar[int]
    int_idx: int
    slice_idx: Slice
    index_list: IndexList
    is_none: bool
    def __init__(self, int_idx: _Optional[int] = ..., slice_idx: _Optional[_Union[Slice, _Mapping]] = ..., index_list: _Optional[_Union[IndexList, _Mapping]] = ..., is_none: bool = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("ndarray", "bool_value", "int_value", "float_value", "str_value", "list_float_value", "list_string_value", "list_bool_value")
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STR_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    ndarray: NDArray
    bool_value: bool
    int_value: int
    float_value: float
    str_value: str
    list_float_value: ListFloat
    list_string_value: ListString
    list_bool_value: ListBool
    def __init__(self, ndarray: _Optional[_Union[NDArray, _Mapping]] = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ..., str_value: _Optional[str] = ..., list_float_value: _Optional[_Union[ListFloat, _Mapping]] = ..., list_string_value: _Optional[_Union[ListString, _Mapping]] = ..., list_bool_value: _Optional[_Union[ListBool, _Mapping]] = ...) -> None: ...

class ListFloat(_message.Message):
    __slots__ = ("list",)
    LIST_FIELD_NUMBER: _ClassVar[int]
    list: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, list: _Optional[_Iterable[float]] = ...) -> None: ...

class ListString(_message.Message):
    __slots__ = ("list",)
    LIST_FIELD_NUMBER: _ClassVar[int]
    list: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, list: _Optional[_Iterable[str]] = ...) -> None: ...

class ListBool(_message.Message):
    __slots__ = ("list",)
    LIST_FIELD_NUMBER: _ClassVar[int]
    list: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, list: _Optional[_Iterable[bool]] = ...) -> None: ...

class Change(_message.Message):
    __slots__ = ("idx", "value")
    IDX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    idx: Idx
    value: Value
    def __init__(self, idx: _Optional[_Union[Idx, _Mapping]] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class Changes(_message.Message):
    __slots__ = ("changes",)
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    changes: _containers.RepeatedCompositeFieldContainer[Change]
    def __init__(self, changes: _Optional[_Iterable[_Union[Change, _Mapping]]] = ...) -> None: ...

class StateChange(_message.Message):
    __slots__ = ("changes", "child")
    class ChildEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StateChange
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StateChange, _Mapping]] = ...) -> None: ...
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    CHILD_FIELD_NUMBER: _ClassVar[int]
    changes: Changes
    child: _containers.MessageMap[str, StateChange]
    def __init__(self, changes: _Optional[_Union[Changes, _Mapping]] = ..., child: _Optional[_Mapping[str, StateChange]] = ...) -> None: ...

class StateChangeList(_message.Message):
    __slots__ = ("state_changes",)
    STATE_CHANGES_FIELD_NUMBER: _ClassVar[int]
    state_changes: _containers.RepeatedCompositeFieldContainer[StateChange]
    def __init__(self, state_changes: _Optional[_Iterable[_Union[StateChange, _Mapping]]] = ...) -> None: ...

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
    def __init__(self, center: _Optional[_Union[NDArray, _Mapping]] = ..., orientation: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ("is_started",)
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...

class Scene(_message.Message):
    __slots__ = ("scene_name",)
    SCENE_NAME_FIELD_NUMBER: _ClassVar[int]
    scene_name: str
    def __init__(self, scene_name: _Optional[str] = ...) -> None: ...
