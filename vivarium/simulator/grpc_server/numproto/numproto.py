"""NumPy ndarray to protobuf serialization and deserialization"""

from io import BytesIO

import numpy as np
from vivarium.simulator.grpc_server.simulator_pb2 import NDArray

# from numproto.protobuf.ndarray_pb2 import NDArray


def ndarray_to_proto(nda: np.ndarray) -> NDArray:
    """Serializes a numpy array into an NDArray protobuf message.

    Args:
        nda (np.ndarray): numpy array to serialize.

    Returns:
        Returns an NDArray protobuf message.
    """
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)

    return NDArray(ndarray=nda_bytes.getvalue())


def proto_to_ndarray(nda_proto: NDArray) -> np.ndarray:
    """Deserializes an NDArray protobuf message into a numpy array.

    Args:
        nda_proto (NDArray): NDArray protobuf message to deserialize.

    Returns:
        Returns a numpy.ndarray.
    """
    nda_bytes = BytesIO(nda_proto.ndarray)

    return np.load(nda_bytes, allow_pickle=False)
