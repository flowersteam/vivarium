import time
import logging
import functools
from threading import Lock
from concurrent import futures
from contextlib import contextmanager


from vivarium.simulator.grpc_server import simulator_pb2_grpc
from vivarium.simulator.grpc_server import simulator_pb2
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc


from vivarium.simulator.grpc_server.converters import dataclass_to_proto, proto_to_changes


lg = logging.getLogger(__name__)


def grpc_error_handler(method):
    """Decorator that catches exceptions in gRPC handlers, logs them, and
    returns a proper gRPC INTERNAL error instead of silently failing."""
    @functools.wraps(method)
    def wrapper(self, request, context):
        try:
            return method(self, request, context)
        except Exception as e:
            lg.exception(f"RPC {method.__name__} failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    return wrapper
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
        
    def _set_changes(self, changes):
        if len(changes) == 0:
            return
        with self._lock:
            with self.simulator.pause():
                self.simulator.set_changes(changes)           
    
    @grpc_error_handler
    def Step(self, request, context):
        self._step()
        return dataclass_to_proto(self.simulator.state)

    @grpc_error_handler
    def SetChangesReturnsState(self, request, context):
        changes = proto_to_changes(request)
        self._set_changes(changes)
        return self.GetStateAndControllerParameters(None, None)

    @grpc_error_handler
    def SetChanges(self, request, context):
        """Apply changes without returning state (for use with streaming)."""
        changes = proto_to_changes(request)
        self._set_changes(changes)
        return Empty()

    @grpc_error_handler
    def SetChangesAndStep(self, request, context):
        changes = proto_to_changes(request)
        self._set_changes(changes)
        self._step()
        return self.GetStateAndControllerParameters(None, None)

    @grpc_error_handler
    def GetState(self, request, context):
        state = self.simulator.state
        p = dataclass_to_proto(state)
        return p

    @grpc_error_handler
    def GetControllerParameters(self, request, context):
        return dataclass_to_proto(self.simulator.controller_parameters)

    @grpc_error_handler
    def GetStateAndControllerParameters(self, request, context):
        state_and_cp = self.simulator.get_state_and_controller_parameters()
        return dataclass_to_proto(state_and_cp)

    @grpc_error_handler
    def GetSceneName(self, request, context):
        scene_name = self.simulator.scene_name
        return simulator_pb2.Scene(scene_name=scene_name)

    @grpc_error_handler
    def RegisterClient(self, request, context):
        lg.info(f"Registering client: {request.name}")
        with self._lock:
            self.simulator.register_client(request.name)
        return Empty()

    @grpc_error_handler
    def UnregisterClient(self, request, context):
        lg.info(f"Unregistering client: {request.name}")
        with self._lock:
            self.simulator.unregister_client(request.name)
        return Empty()

    @grpc_error_handler
    def Start(self, request, context):
        self.simulator.run(threaded=True)
        return Empty()

    @grpc_error_handler
    def IsRunning(self, request, context):
        return simulator_pb2.IsRunningState(is_running=self.simulator.is_running())

    @grpc_error_handler
    def Stop(self, request, context):
        self.simulator.stop()
        return Empty()

    # ============ Streaming RPCs ============

    def StreamState(self, request, context):
        """Server-side streaming: Push state updates to client.

        Used when server is running continuously and client wants to observe.
        Yields state updates at the configured FPS rate.
        """
        max_fps = request.max_fps if request.max_fps > 0 else 60
        min_interval = 1.0 / max_fps
        include_cp = request.include_controller_params

        lg.info(f"StreamState started (max_fps={max_fps}, include_cp={include_cp})")

        try:
            while context.is_active():
                start_time = time.time()

                # Get current state
                if include_cp:
                    state_and_cp = self.simulator.get_state_and_controller_parameters()
                    yield dataclass_to_proto(state_and_cp)
                else:
                    yield dataclass_to_proto(self.simulator.state)

                # Rate limiting - sleep for remaining time in interval
                elapsed = time.time() - start_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
        except Exception as e:
            lg.exception("RPC StreamState failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        lg.info("StreamState ended")


def create_grpc_server(simulator, port=50051):
    """
    Create a gRPC server with the simulator servicer and health checking.

    Args:
        simulator: The Simulator instance to serve
        port: Port to listen on (use 0 for random available port)

    Returns:
        tuple: (server, actual_port) - the gRPC server and the port it's listening on

    Raises:
        RuntimeError: If port binding fails
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server
    )

    # Add health checking service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    actual_port = server.add_insecure_port(f"[::]:{port}")
    if actual_port == 0:
        raise RuntimeError(
            f"Failed to bind to port {port}. "
            "Another process may be using it. Try: lsof -i :50051"
        )

    server.start()
    lg.info(f"gRPC server started on port {actual_port}")

    # Mark service as ready
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    lg.info("gRPC health service marked as SERVING")

    return server, actual_port


def serve(simulator):
    server, port = create_grpc_server(simulator, port=50051)
    lg.info(f"Server listening on port {port}, waiting for termination...")
    server.wait_for_termination()
