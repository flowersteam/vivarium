import logging
from contextlib import contextmanager

import param

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.config import SimulatorConfig
from vivarium.simulator.simulator_states import StateType
from vivarium.controllers import converters

lg = logging.getLogger(__name__)

param.Dynamic.time_dependent = True


class SimulatorController(param.Parameterized):
    """Base controller class to interact with the simulator."""

    configs = param.Dict(
        {
            StateType.SIMULATOR: SimulatorConfig(),
            StateType.AGENT: [],
            StateType.OBJECT: [],
        }
    )
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, client=None, **params):
        super().__init__(**params)
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        configs_dict = converters.set_configs_from_state(self.state)
        for stype, configs in configs_dict.items():
            self.configs[stype] = configs
        self._config_watchers = self.watch_configs()
        self.pull_all_data()
        self.client.name = self.name
        self._in_batch = False
        self.scene_name = self.client.scene_name
        self.subtypes_labels = self.client.subtypes_labels

    def watch_configs(self):
        """Watch the parameters of the configs to push the changes to the simulator."""
        watchers = {
            etype: [
                config.param.watch(
                    self.push_state, config.param_names(), onlychanged=True
                )
                for config in configs
            ]
            for etype, configs in self.configs.items()
        }
        return watchers

    @property
    def simulator_config(self):
        return self.configs[StateType.SIMULATOR][0]

    def push_state(self, *events):
        """Push the state changes to the simulator."""
        if self._in_batch:
            self._event_list.extend(events)
            return
        # lg.debug("Push_state %d", len(events))
        state_changes = converters.events_to_state_changes(events, self.state)
        for sc in state_changes:
            self.client.set_state(**sc._asdict())

    @contextmanager
    def dont_push_entity_configs(self):
        """Context manager to avoid pushing the entity configs to the simulator."""
        for etype, configs in self.configs.items():
            for i, config in enumerate(configs):
                config.param.unwatch(self._config_watchers[etype][i])
        try:
            yield None
        finally:
            self._config_watchers = self.watch_configs()

    @contextmanager
    def batch_set_state(self):
        """Context manager to set the state changes in batch."""
        self._in_batch = True
        self._event_list = []
        try:
            yield
        finally:
            self._in_batch = False
            self.push_state(*self._event_list)
            self._event_list = None

    def pull_all_data(self):
        """Pull all the data from the simulator."""
        self.pull_configs()

    def pull_configs(self, configs=None):
        """Pull the configurations."""
        configs = configs or self.configs
        state = self.state
        with self.dont_push_entity_configs():
            converters.set_configs_from_state(state, configs)
        return state

    def is_started(self):
        """Check if the simulator is started."""
        return self.client.is_started()

    def start(self):
        """Start the simulator."""
        self.client.start()

    def stop(self):
        """Stop the simulator."""
        self.client.stop()

    def update_state(self):
        """Update the state of the simulator."""
        self.state = self.client.get_state()
        return self.state

    def get_nve_state(self):
        """Get the NVE state of the simulator."""
        self.state = self.client.get_nve_state()
        return self.state

    def get_scene_name(self):
        """Get the scene name of the simulator."""
        self.client.get_scene_name()

    def get_subtypes_labels(self):
        """Get the subtypes labels of the simulator."""
        self.client.get_subtypes_labels()
