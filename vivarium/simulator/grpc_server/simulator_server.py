import logging

from concurrent import futures
from threading import Lock
from contextlib import contextmanager
from collections import defaultdict

import grpc
import simulator_pb2
import simulator_pb2_grpc

from numproto.numproto import proto_to_ndarray

from vivarium.simulator.grpc_server.converters import state_to_proto
from vivarium.simulator.grpc_server.converters import nve_state_to_proto
from vivarium.simulator.grpc_server.converters import agent_state_to_proto
from vivarium.simulator.grpc_server.converters import object_state_to_proto


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
    def __init__(self, simulator):
        self.simulator = simulator
        self.recorded_change_dict = defaultdict(dict)
        self._change_time = 0
        self._simulation_time = 0
        self._lock = Lock()

    def GetState(self, request, context):
        state = self.simulator.state
        return state_to_proto(state)

    def GetNVEState(self, request, context):
        entities_state = self.simulator.state.entities_state
        return nve_state_to_proto(entities_state)

    def GetAgentState(self, request, context):
        agent_state = self.simulator.state.agent_state
        return agent_state_to_proto(agent_state)

    def GetObjectState(self, request, context):
        object_state = self.simulator.state.object_state
        return object_state_to_proto(object_state)

    def Start(self, request, context):
        self.simulator.run(threaded=True)
        return Empty()

    def IsStarted(self, request, context):
        return simulator_pb2.IsStartedState(is_started=self.simulator.is_started())
    
    def LoadScene(self, request, context):
        with self._lock:
            self.simulator.load_scene(request.scene)
        return Empty()

    def Stop(self, request, context):
        self.simulator.stop()
        return Empty()

    def SetState(self, request, context):

        with self._lock:
            nve_idx = request.nve_idx
            col_idx = request.col_idx
            # with self.simulator.pause():
            self.simulator.set_state(request.nested_field, nve_idx, col_idx,
                                                   proto_to_ndarray(request.value))
        return Empty()

    def Step(self, request, context):
        assert not self.simulator.is_started()
        self.simulator.step()
        return state_to_proto(self.simulator.state)


def serve(simulator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
