from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.dataclass_wrapper import EntityList, DataclassWrapper
from vivarium.simulator.simulator_states import EntityType


class ClientDataclassWrapper(DataclassWrapper):
    def __init__(self, client):
        super().__init__()
        self._client = client

    def apply(self):
        self._client.apply_changes([self.fetch_changes()])


class SimulatorController:
    def __init__(self, client=None):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.create_entity_list()

    def create_entity_list(self):
        self.entity_lists = {
            etype: EntityList(
                state=self.state, entity_type=etype
            )
            for etype in EntityType
        }

    @property
    def agents(self):
        return self.entity_lists[EntityType.AGENT]
    
    @property
    def objects(self):
        return self.entity_lists[EntityType.OBJECT]

    def step(self):
        changes = self.fetch_entity_lists_changes()
        self.state = self.client.step(changes)
        self.update_entity_lists()

    def update_entity_lists(self, state=None):
        """Update the entity lists."""
        state = state or self.state
        for _, ent_list in self.entity_lists.items():
            ent_list.set_state(state)

    def update_state(self):
        """Update the state of the simulator."""
        self.state = self.client.get_state()
        self.update_entity_lists()
        return self.state

    def fetch_entity_lists_changes(self):
        changes = []
        for etype, elist in self.entity_lists.items():
            changes.extend(elist.fetch_changes())
        return changes

    def apply_changes(self):
        changes = self.fetch_entity_lists_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)
