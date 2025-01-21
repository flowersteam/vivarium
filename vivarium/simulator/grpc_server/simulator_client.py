import grpc
from numproto.numproto import ndarray_to_proto
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.grpc_server import simulator_pb2_grpc
from vivarium.simulator.grpc_server.simulator_client_abc import SimulatorClient
from vivarium.simulator.grpc_server.converters import (
    proto_to_state,
    proto_to_nve_state,
    proto_to_agent_state,
    proto_to_object_state,
)

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorGRPCClient(SimulatorClient):
    """A client for the simulator server that uses gRPC.

    :param SimulatorClient: Abstract base class for simulator clients.
    """

    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel("localhost:50051")
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)
        self.streaming_started = False
        self.state = self.get_state()
        self.scene_name = self.get_scene_name()
        self.subtypes_labels = self.get_subtypes_labels()

    def start(self):
        """Start the simulator."""
        self.stub.Start(Empty())

    def stop(self):
        """Stop the simulator."""
        self.stub.Stop(Empty())

    def get_change_time(self):
        """Get the change time of the simulator."""
        return self.stub.GetChangeTime(Empty()).time

    def set_state(self, nested_field, ent_idx, column_idx, value):
        """Set the state of the simulator.

        :param nested_field: nested field to set
        :param ent_idx: entity index to set
        :param column_idx: column index to set
        :param value: value to set
        """
        state_change = simulator_pb2.StateChange(
            nested_field=nested_field,
            ent_idx=ent_idx,
            col_idx=column_idx,
            value=ndarray_to_proto(value),
        )
        self.stub.SetState(state_change)

    def get_state(self):
        """Get the state of the simulator.

        :return: simulation state
        """
        state = self.stub.GetState(Empty())
        return proto_to_state(state)

    def get_nve_state(self):
        """Get the NVE state of the simulator.

        :return: simulation Entity state
        """
        entity_state = self.stub.GetNVEState(Empty())
        return proto_to_nve_state(entity_state)

    def get_agent_state(self):
        """Get the agent state of the simulator.

        :return: simulation Agent state
        """
        agent_state = self.stub.GetAgentState(Empty())
        return proto_to_agent_state(agent_state)

    def get_object_state(self):
        """Get the object state of the simulator.

        :return: simulation Object state
        """
        object_state = self.stub.GetObjectState(Empty())
        return proto_to_object_state(object_state)

    def get_scene_name(self):
        """Get the scene name of the simulator.

        :return: scene name
        """
        response = self.stub.GetSceneName(Empty())
        scene_name = response.scene_name
        return scene_name

    def get_subtypes_labels(self):
        """Get the subtypes labels of the simulator.

        :return: subtypes labels
        """
        response = self.stub.GetSubtypesLabels(Empty())
        subtype_labels_dict = dict(response.data)
        return subtype_labels_dict

    def step(self):
        """Step the simulator.

        :return: simulation state
        """
        self.state = proto_to_state(self.stub.Step(Empty()))
        return self.state

    def is_started(self):
        """Check if the simulator is started."""
        return self.stub.IsStarted(Empty()).is_started
