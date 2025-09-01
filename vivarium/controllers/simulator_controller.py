import hydra
from dataclasses import asdict

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import SimulatorParametersWrapper


class SimulatorController:

    def __init__(self, client=None, subtypes=[], **controllers):
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        self.simulator_parameters = self.client.get_simulator_parameters()
        self.subtype_labels = {i: label for i, label in enumerate(subtypes)}
        
        self.controllers = controllers
        
        self.create_simulator_parameters_wrapper()

    @classmethod
    def from_config(cls, config, client=None, notebook_control=False):
        controllers = {}
        state = client.state
        cp = asdict(client.controller_parameters)
        for name, c_config in config.component_list.items():
            if 'client' in c_config:
                c_cls = hydra.utils.get_class(c_config.client.controller_cls)
                p = {} if cp[name] is None else cp[name]
                controllers[name] = c_cls.from_config(name, c_config.client, state, notebook_control=notebook_control, **p)
        return cls(
            subtypes=config.subtype_labels,
            client=client,
            **controllers
        )


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
        self.update_controllers()

    def update_controllers(self, state=None):
        """Update the controllers."""
        state = state or self.state
        for _, controller in self.controllers.items():
            controller.set_state(state)

    def update_state(self):
        """Update the state from server to client."""
        self.state = self.client.get_state()
        self.update_controllers()
        return self.state

    def fetch_changes(self):
        changes = []
        for _, controller in self.controllers.items():
            change = controller.fetch_changes()
            changes.extend(change)

        change = self.simulator_parameters.fetch_changes()
        if change:
            changes.extend([change])
        return changes

    def apply_changes(self):
        changes = self.fetch_changes()
        if len(changes) > 0:
            self.client.apply_changes(changes)
