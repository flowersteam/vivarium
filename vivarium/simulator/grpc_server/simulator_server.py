import logging
from threading import Lock
from concurrent import futures
from collections import defaultdict
from contextlib import contextmanager


import simulator_pb2_grpc
import simulator_pb2
import grpc

from numproto.numproto import proto_to_ndarray

from vivarium.simulator.grpc_server.converters import dataclass_to_proto, proto_to_dataclass, proto_to_changes
from vivarium.simulator.config import SimulatorConfiguration


lg = logging.getLogger(__name__)
Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty



@contextmanager
def nonblocking(lock):
    locked = lock.acquire(False)
    try:
        yield locked
    finally:
        if locked:
            lock.release()

class SimulatorServerServicer(simulator_pb2_grpc.SimulatorServerServicer):
    """A gRPC server for the simulator.

    :param simulator_pb2_grpc: The gRPC server for the simulator.
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.simulator.init_state()
        self.recorded_change_dict = defaultdict(dict)
        self._lock = Lock()

    def SetChanges(self, request, context):
        changes = proto_to_changes(request)
        with self._lock:
            self.simulator.apply_changes(changes)
        return self.GetControllerParameters(None, None)

    def SetChangesAndStep(self, request, context):
        proto_cp = self.SetChanges(request, context)
        proto_state = self.Step(None, None)
        proto_dataclass = simulator_pb2.Dataclass()
        proto_dataclass.nested_fields['controller_parameters'].CopyFrom(proto_cp)
        proto_dataclass.nested_fields['state'].CopyFrom(proto_state)
        return proto_dataclass

    def GetState(self, request, context):
        state = self.simulator.state
        p = dataclass_to_proto(state)
        return p
    
    def GetSimulatorParameters(self, request, context):
        parameters = SimulatorConfiguration.from_simulator(self.simulator)
        return dataclass_to_proto(parameters)
    
    def GetControllerParameters(self, request, context):
        return dataclass_to_proto(self.simulator.controller_parameters)

    def GetSceneName(self, request, context):
        scene_name = self.simulator.scene_name
        return simulator_pb2.Scene(scene_name=scene_name)

    def SetControllerAtrributes(self, request, context):
        with self._lock:
            self.simulator.controller_parameters = proto_to_dataclass(request)
        return Empty()
    
    def Start(self, request, context):
        self.simulator.run(threaded=True)
        return Empty()

    def IsStarted(self, request, context):
        return simulator_pb2.IsStartedState(is_started=self.simulator.is_started())

    def Stop(self, request, context):
        self.simulator.stop()
        return Empty()

    def SetState(self, request, context):
        with self._lock:
            ent_idx = request.ent_idx
            col_idx = request.col_idx
            self.simulator.set_state(
                request.nested_field, ent_idx, col_idx, proto_to_ndarray(request.value)
            )
        return Empty()

    def Step(self, request, context):
        assert not self.simulator.is_started()
        self.simulator.step()
        return dataclass_to_proto(self.simulator.state)


def serve(simulator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
