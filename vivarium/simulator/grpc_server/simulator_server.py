import logging
from threading import Lock
from concurrent import futures
from collections import defaultdict
from contextlib import contextmanager


from numproto.numproto import proto_to_ndarray
import simulator_pb2_grpc
import simulator_pb2
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc


from vivarium.simulator.grpc_server.converters import dataclass_to_proto, proto_to_dataclass, proto_to_changes


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
        self._lock = Lock()
    
    def _step(self):
        assert not self.simulator.is_running()
        self.simulator.step()
        
    def _apply_changes(self, changes):
        if len(changes) == 0:
            return
        with self._lock:
            with self.simulator.pause():
                self.simulator.apply_changes(changes)           
    
    def Step(self, request, context):
        self._step()
        return dataclass_to_proto(self.simulator.state)
    
    def SetChanges(self, request, context):
        changes = proto_to_changes(request)
        self._apply_changes(changes)
        return self.GetStateAndControllerParameters(None, None)

    def SetChangesAndStep(self, request, context):
        changes = proto_to_changes(request)
        self._apply_changes(changes)
        self._step()
        return self.GetStateAndControllerParameters(None, None)

    def GetState(self, request, context):
        state = self.simulator.state
        p = dataclass_to_proto(state)
        return p
    
    def GetControllerParameters(self, request, context):
        return dataclass_to_proto(self.simulator.controller_parameters)
    
    def GetStateAndControllerParameters(self, request, context):
        state_and_cp = self.simulator.get_state_and_controller_parameters()
        return dataclass_to_proto(state_and_cp)

    def GetSceneName(self, request, context):
        scene_name = self.simulator.scene_name
        return simulator_pb2.Scene(scene_name=scene_name)
    
    def RegisterClient(self, request, context):
        lg.info(f"Registering client: {request.name}")
        with self._lock:
            self.simulator.register_client(request.name)
        return Empty()

    def UnregisterClient(self, request, context):
        lg.info(f"Unregistering client: {request.name}")
        with self._lock:
            self.simulator.unregister_client(request.name)
        return Empty()

    def Start(self, request, context):
        self.simulator.run(threaded=True)
        return Empty()

    def IsRunning(self, request, context):
        return simulator_pb2.IsRunningState(is_running=self.simulator.is_running())

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


def serve(simulator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server
    )
    
    # Add health checking service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    server.add_insecure_port("[::]:50051")
    server.start()
    
    # Mark service as ready
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    
    server.wait_for_termination()
