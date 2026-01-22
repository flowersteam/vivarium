import grpc
import uuid
import time
import logging
import threading
from hydra.utils import get_class
from dataclasses import dataclass

from vivarium.simulator.grpc_server.converters import proto_to_dataclass, changes_to_proto
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2
from vivarium.simulator.grpc_server import simulator_pb2_grpc

from vivarium.environment.state import create_state_cls
from vivarium.utils.dataclass_wrapper import Remote
from vivarium.utils.scene_configs import load_scene_config
from vivarium.utils.scene_configs import component_factories_from_config


Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty

lg = logging.getLogger(__name__)


# @access_nested_fields(nested_fields_to_access)
class SimulatorGRPCClient:
    """A client for the simulator server that uses gRPC.
    """

    def __init__(self, name=None, server=None):
        self.name = name if name is not None else str(uuid.uuid4())
        self.channel = grpc.insecure_channel(server or "localhost:50051")
        self.stub = simulator_pb2_grpc.SimulatorServerStub(self.channel)
        self.register_client(self.name)
        config = load_scene_config(self.scene_name)
        update_fns = [f.update_state_cls for f in component_factories_from_config(config.environment.components)]
        self.state_cls = create_state_cls(
            base_state_cls=get_class(config.environment.kwargs.base_state_cls),
            update_fns=update_fns
        )
        self.state = self.get_state()
        self.controller_parameters = self.get_controller_parameters()
        
        @dataclass
        class StateAndControllerParameters:
            state: self.state_cls
            controller_parameters: type(self.controller_parameters)
        self.state_and_cp_cls = StateAndControllerParameters
        
        self.remote = Remote(self)
        
        # Streaming state
        self._stream_thread = None
        self._stream_stop_event = None

    @property
    def is_streaming(self):
        """Check if state streaming is currently active."""
        return self._stream_thread is not None and self._stream_thread.is_alive()

    def set_changes(self, changes, update_from_server=True):
        """Apply changes to the simulator server
        Args:
            changes: list of changes to apply
            update_from_server: whether to request the state and controller parameters from the server and update them here.
            Disable it when state updates are already being received via streaming,
        to avoid redundant state serialization (typically in WindowManager).
        """
        if self.channel is None:
            lg.warning("Channel is closed, cannot set changes.")
            return
        proto_changes = changes_to_proto(changes)
        if update_from_server:
            state_and_cp = proto_to_dataclass(self.stub.SetChangesReturnsState(proto_changes), self.state_and_cp_cls)
            self.state = state_and_cp.state
            self.controller_parameters = state_and_cp.controller_parameters

        else:
            proto_changes = changes_to_proto(changes)
            self.stub.SetChanges(proto_changes)

    def start(self):
        """Start the simulator."""
        self.stub.Start(Empty())

    def stop(self):
        """Stop the simulator."""
        self.stub.Stop(Empty())

    def get_state(self):
        """Get the state of the simulator.

        :return: simulation state
        """
        state = self.stub.GetState(Empty())
        return proto_to_dataclass(state, self.state_cls)

    def get_controller_parameters(self):
        """Get the controller parameters of the simulator.

        :return: controller parameters
        """
        parameters = self.stub.GetControllerParameters(Empty())
        return proto_to_dataclass(parameters)

    @property
    def scene_name(self):
        """Get the scene name of the simulator.

        :return: scene name
        """
        response = self.stub.GetSceneName(Empty())
        scene_name = response.scene_name
        return scene_name

    def step(self, changes=None):
        """Step the simulator.

        :return: simulation state
        """
        res = proto_to_dataclass(self.stub.SetChangesAndStep(changes_to_proto(changes)), self.state_and_cp_cls)
        self.state = res.state
        self.controller_parameters = res.controller_parameters
        return res

    def is_running(self):
        """Check if the simulator is started."""
        return self.stub.IsRunning(Empty()).is_running
    
    def register_client(self, name):
        """Register a client with the simulator."""
        self.stub.RegisterClient(simulator_pb2.Client(name=name))
        
    def unregister_client(self, name):
        """Unregister a client from the simulator."""
        self.stub.UnregisterClient(simulator_pb2.Client(name=name))
        
    def close(self):
        """Close the gRPC channel."""
        self.unregister_client(self.name)
        self.stop_state_stream(blocking=True)
        self.channel.close()
        self.channel = None

    # ============ Streaming Methods ============

    def start_state_stream(self, callback, max_fps=30, include_controller_params=True):
        """Start receiving state updates via server-side streaming.
        
        Args:
            callback: Function called with each state update (receives state or state_and_cp)
            max_fps: Maximum updates per second
            include_controller_params: Whether to include controller parameters
            
        Returns:
            Thread object running the stream
        """
        if self._stream_thread is not None:
            self.stop_state_stream()
        
        self._stream_stop_event = threading.Event()
        
        def stream_worker():
            config = simulator_pb2.StreamConfig(
                max_fps=max_fps,
                include_controller_params=include_controller_params
            )
            try:
                for proto_state in self.stub.StreamState(config):
                    if self._stream_stop_event.is_set():
                        break
                    if include_controller_params:
                        state_and_cp = proto_to_dataclass(proto_state, self.state_and_cp_cls)
                        self.state = state_and_cp.state
                        self.controller_parameters = state_and_cp.controller_parameters
                        callback(state_and_cp)
                    else:
                        state = proto_to_dataclass(proto_state, self.state_cls)
                        self.state = state
                        callback(state)
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.CANCELLED:
                    raise
        
        self._stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self._stream_thread.start()
        return self._stream_thread

    def stop_state_stream(self, blocking=True):
        """Stop the state streaming."""
        if self._stream_stop_event is not None:
            self._stream_stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
            self._stream_stop_event = None
        
        lg.info("Stopping state stream ...")
        while blocking and self.is_streaming:
            time.sleep(0.1)

    def bidirectional_step_generator(self, changes_iterator):
        """Generator for bidirectional stepping.
        
        Args:
            changes_iterator: Iterator yielding changes to send
            
        Yields:
            State updates from server after each step
            
        Note: This is NOT synchronized - the changes_iterator runs independently
        of responses. For synchronized stepping, use bidirectional_step_sync().
        """
        def request_generator():
            for changes in changes_iterator:
                yield changes_to_proto(changes)
        
        for proto_state in self.stub.BidirectionalStep(request_generator()):
            state_and_cp = proto_to_dataclass(proto_state, self.state_and_cp_cls)
            self.state = state_and_cp.state
            self.controller_parameters = state_and_cp.controller_parameters
            yield state_and_cp

    def bidirectional_step_sync(self, num_steps, compute_changes_fn):
        """Synchronized bidirectional stepping with proper state synchronization.
        
        This maintains a persistent stream but ensures each step waits for the
        previous state before computing the next motor commands.
        
        Args:
            num_steps: Number of steps to execute (can be math.inf for infinite)
            compute_changes_fn: Function that computes and returns changes.
                                Can have signature:
                                - compute_changes_fn() -> changes_list  (uses self.state internally)
                                - compute_changes_fn(state) -> changes_list
                                Returns None or raises StopIteration to stop the loop.
                                
        Yields:
            State updates from server after each step
            
        Example:
            def compute_motors():
                # Access client.state directly
                # Compute motor commands based on current state
                return []  # or list of changes, or None to stop
            
            for state_and_cp in client.bidirectional_step_sync(100, compute_motors):
                pass  # State already updated in client
        """
        import inspect
        
        # Check if compute_changes_fn takes a state argument
        sig = inspect.signature(compute_changes_fn)
        takes_state = len(sig.parameters) > 0
        
        # Use threading events to synchronize request/response
        state_ready = threading.Event()
        stop_flag = threading.Event()
        steps_done = [0]
        
        def synchronized_request_generator():
            while steps_done[0] < num_steps and not stop_flag.is_set():
                # Compute changes (state is already updated in self.state)
                try:
                    if takes_state:
                        changes = compute_changes_fn(self.state)
                    else:
                        changes = compute_changes_fn()
                except StopIteration:
                    break
                    
                # None signals stop
                if changes is None:
                    break
                    
                yield changes_to_proto(changes)
                steps_done[0] += 1
                
                # Wait for state to be updated before computing next changes
                # (except for the last iteration)
                if steps_done[0] < num_steps:
                    state_ready.wait(timeout=30.0)
                    if stop_flag.is_set():
                        break
                    state_ready.clear()
        
        try:
            for proto_state in self.stub.BidirectionalStep(synchronized_request_generator()):
                state_and_cp = proto_to_dataclass(proto_state, self.state_and_cp_cls)
                self.state = state_and_cp.state
                self.controller_parameters = state_and_cp.controller_parameters
                
                # Signal that state is ready for next computation
                state_ready.set()
                
                yield state_and_cp
        finally:
            stop_flag.set()
            state_ready.set()  # Unblock generator if waiting
            lg.info('bidirectional_step_sync stopped')
