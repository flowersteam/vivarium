import grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.grpc_server import simulator_pb2_grpc
from vivarium.simulator.grpc_server.simulator_client_abc import SimulatorClient
from vivarium.simulator.grpc_server.converters import (
    proto_to_state,
    changes_to_proto
)
from vivarium.simulator.simulator_states import SimState

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorGRPCClient(SimulatorClient):
    """A client for the simulator server that uses gRPC.

    :param SimulatorClient: Abstract base class for simulator clients.
    """

    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel("localhost:50051")
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)
        self.state = self.get_state()
        self.scene_name = self.get_scene_name()
        self.subtypes_labels = self.get_subtype_labels()

    def apply_changes(self, changes):
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
        return proto_to_state(state, SimState)

    def get_scene_name(self):
        """Get the scene name of the simulator.

        :return: scene name
        """
        response = self.stub.GetSceneName(Empty())
        scene_name = response.scene_name
        return scene_name

    def get_subtype_labels(self):
        """Get the subtypes labels of the simulator.

        :return: subtypes labels
        """
        response = self.stub.GetSubtypesLabels(Empty())
        subtype_labels_dict = dict(response.data)
        return subtype_labels_dict

    def step(self, changes=[]):
        """Step the simulator.

        :return: simulation state
        """
        if len(changes) > 0:
            self.state = proto_to_state(self.stub.SetChangesAndStep(changes_to_proto(changes)), SimState)
        else:
            self.state = proto_to_state(self.stub.Step(Empty()), SimState)
        return self.state

    def is_started(self):
        """Check if the simulator is started."""
        return self.stub.IsStarted(Empty()).is_started
