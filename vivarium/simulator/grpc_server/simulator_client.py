import grpc
from hydra.utils import get_class
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2
from vivarium.simulator.grpc_server.simulator_client_abc import SimulatorClient
from vivarium.simulator.grpc_server.converters import proto_to_dataclass, changes_to_proto

from vivarium.utils.scene_configs import load_scene_config
from vivarium.utils.scene_configs import component_factories_from_config
from vivarium.environment.state import create_state_cls


Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


# @access_nested_fields(nested_fields_to_access)
class SimulatorGRPCClient(SimulatorClient):
    """A client for the simulator server that uses gRPC.
    """

    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel("localhost:50051")
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)
        config = load_scene_config(self.scene_name)
        update_fns = [f.update_state_cls for f in component_factories_from_config(config.environment.components)]
        self.state_cls = create_state_cls(
            base_state_cls=get_class(config.environment.kwargs.base_state_cls),
            update_fns=update_fns
        )
        self.state = self.get_state()

    def apply_changes(self, changes):
        proto_changes = changes_to_proto(changes)
        controller_parameters = proto_to_dataclass(self.stub.SetChanges(proto_changes))
        return controller_parameters

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
    
    def get_simulator_parameters(self):
        parameters = self.stub.GetSimulatorParameters(Empty())
        return proto_to_dataclass(parameters) #, SimulatorConfiguration)

    @property
    def controller_parameters(self):
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


    def step(self, changes=[]):
        """Step the simulator.

        :return: simulation state
        """
        if len(changes) > 0:
            # res will be a dataclass with fields state and controller_parameters
            res = proto_to_dataclass(self.stub.SetChangesAndStep(changes_to_proto(changes)))
            self.state = res.state
        else:
            # res will be the state
            res = proto_to_dataclass(self.stub.Step(Empty()), self.state_cls)
            self.state = res
        return res

    def is_started(self):
        """Check if the simulator is started."""
        return self.stub.IsStarted(Empty()).is_started
