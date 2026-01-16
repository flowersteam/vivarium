import grpc
import uuid
from hydra.utils import get_class
from dataclasses import dataclass

from vivarium.simulator.grpc_server.converters import proto_to_dataclass, changes_to_proto
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2
from vivarium.simulator.grpc_server import simulator_pb2_grpc

from vivarium.environment.state import create_state_cls
from vivarium.controllers.dataclass_wrapper import Remote
from vivarium.utils.scene_configs import load_scene_config
from vivarium.utils.scene_configs import component_factories_from_config


Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


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

    def apply_changes(self, changes):
        proto_changes = changes_to_proto(changes)
        state_and_cp = proto_to_dataclass(self.stub.SetChanges(proto_changes), self.state_and_cp_cls)
        self.state = state_and_cp.state
        self.controller_parameters = state_and_cp.controller_parameters

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
        self.channel.close()
