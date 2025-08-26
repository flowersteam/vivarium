import hydra
from dataclasses import asdict

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.dataclass_wrapper import SimulatorParametersWrapper
from vivarium.controllers.panel_controller import ParamSimulator


class SimulatorController:

    def __init__(self, client=None, subtypes=[], **controllers):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.simulator_parameters = self.client.get_simulator_parameters()
        self.subtype_labels = {i: label for i, label in enumerate(subtypes)}
        
        self.controllers = controllers

        # TODO: (2025-08-26) move this to a dedicated class?
        self.param_simulator = ParamSimulator(self.simulator_parameters)
        self.param_simulator.update_from_server = True
        
        self.entity_lists = self.create_entity_list()

        for etype, elist in self.entity_lists.items():
            setattr(self, etype, elist)
        self.create_simulator_parameters_wrapper()

    @classmethod
    def from_config(cls, config, client=None):
        controllers = {}
        state = client.state
        cp = asdict(client.controller_parameters)
        for etype, e_config in config.component_list.items():
            if 'client' in e_config:
                e_cls = hydra.utils.get_class(e_config.client.controller_cls)
                controllers[etype] = e_cls(etype, state, config.subtype_labels, **cp[etype])
        return cls(
            subtypes=config.subtype_labels,
            client=client,
            **controllers
        )

    def create_entity_list(self):
        return {etype: c.controller for etype, c in self.controllers.items()}

    def create_simulator_parameters_wrapper(self):
        self.simulator_parameters = SimulatorParametersWrapper(self.simulator_parameters)

    def start(self):
        """Start the simulator."""
        self.client.start()

    def stop(self):
        """Stop the simulator."""
        self.client.stop()

    def is_started(self):
        """Check if the simulator is started."""
        return self.client.is_started()

    def step(self):
        changes = self.fetch_changes()
        self.state = self.client.step(changes)
        self.update_entity_lists()

    def update_entity_lists(self, state=None):
        """Update the entity lists."""
        state = state or self.state
        for _, ent_list in self.entity_lists.items():
            ent_list.set_state(state)

    def update_state(self):
        """Update the state from server to client."""
        self.state = self.client.get_state()
        self.update_entity_lists()
        return self.state

    def fetch_changes(self):
        changes = []
        for etype, elist in self.entity_lists.items():
            change = elist.fetch_changes()
            change = [{'state': c} for c in change]
            changes.extend(change)
            for e in elist:
                change = e._controller_change_recorder.fetch_changes()
                if change:
                    changes.extend([{'controller_parameters': {etype: change}}])
        change = self.simulator_parameters.fetch_changes()
        if change:
            changes.extend([change])
        return changes

    def apply_changes(self):
        changes = self.fetch_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)
