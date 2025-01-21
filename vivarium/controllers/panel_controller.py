import time
import threading
import param
import logging
import numpy as np
from contextlib import contextmanager

from vivarium.controllers import converters
from vivarium.controllers.config import (
    AgentConfig,
    ObjectConfig,
    config_to_stype,
    Config,
)
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.simulator_states import EntityType, StateType
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient

lg = logging.getLogger(__name__)


class PanelConfig(Config):
    """Base class for panel configurations"""

    pass


class PanelEntityConfig(PanelConfig):
    """Base class for panel configurations of entities"""

    visible = param.Boolean(True)


class PanelAgentConfig(PanelEntityConfig):
    """Base class for panel configurations of agents"""

    visible_wheels = param.Boolean(True)
    visible_proxs = param.Boolean(True)


class PanelObjectConfig(PanelEntityConfig):
    """Base class for panel configurations of objects"""

    pass


class PanelSimulatorConfig(Config):
    """Base class for panel configurations of the simulator"""

    hide_non_existing = param.Boolean(True)
    config_update = param.Boolean(False)


# Mapping between config classes and their corresponding state types
panel_config_to_stype = {
    PanelSimulatorConfig: StateType.SIMULATOR,
    PanelAgentConfig: StateType.AGENT,
    PanelObjectConfig: StateType.OBJECT,
}

stype_to_panel_config = {
    stype: config_class for config_class, stype in panel_config_to_stype.items()
}


class Selected(param.Parameterized):
    """Class to store the selected entities in the interface"""

    selection = param.ListSelector([0], objects=[0])

    def selection_nve_idx(self, ent_idx):
        return ent_idx[np.array(self.selection)].tolist()

    def __len__(self):
        return len(self.selection)


class PanelController(SimulatorController):
    """Controller for the panel interface"""

    def __init__(self, **params):
        self._selected_configs_watchers = None
        self._selected_panel_configs_watchers = None
        self.selected_entities = {
            EntityType.AGENT: Selected(),
            EntityType.OBJECT: Selected(),
        }
        self.selected_configs = {
            EntityType.AGENT: AgentConfig(),
            EntityType.OBJECT: ObjectConfig(),
        }
        super().__init__(**params)
        self.panel_configs = {
            stype: [stype_to_panel_config[stype]() for _ in range(len(configs))]
            for stype, configs in self.configs.items()
        }
        self.selected_panel_configs = {
            EntityType.AGENT: PanelAgentConfig(),
            EntityType.OBJECT: PanelObjectConfig(),
        }
        self.panel_simulator_config = PanelSimulatorConfig()
        self.pull_selected_panel_configs()

        self.update_entity_list()
        for selected in self.selected_entities.values():
            selected.param.watch(
                self.pull_selected_configs,
                ["selection"],
                onlychanged=True,
                precedence=1,
            )
            selected.param.watch(
                self.pull_selected_panel_configs, ["selection"], onlychanged=True
            )
        # Add this to force non existing entities to be hidden at the initialization of the interface
        threading.Timer(1.0, self.trigger_hide_non_existing).start()

    def trigger_hide_non_existing(self):
        """Triggers the hide_non_existing parameter change"""
        self.panel_simulator_config.hide_non_existing = False
        time.sleep(0.1)
        self.panel_simulator_config.hide_non_existing = True

    def watch_selected_configs(self):
        """Watch the selected configurations"""
        watchers = {
            etype: config.param.watch(
                self.push_selected_to_config_list,
                config.param_names(),
                onlychanged=True,
            )
            for etype, config in self.selected_configs.items()
        }
        return watchers

    def watch_selected_panel_configs(self):
        """Watch the selected panel configurations"""
        watchers = {
            etype: config.param.watch(
                self.push_selected_to_config_list,
                config.param_names(),
                onlychanged=True,
            )
            for etype, config in self.selected_panel_configs.items()
        }
        return watchers

    @contextmanager
    def dont_push_selected_configs(self):
        """Context manager to avoid pushing the selected configurations"""
        if self._selected_configs_watchers is not None:
            for etype, config in self.selected_configs.items():
                config.param.unwatch(self._selected_configs_watchers[etype])
        try:
            yield
        finally:
            self._selected_configs_watchers = self.watch_selected_configs()

    @contextmanager
    def dont_push_selected_panel_configs(self):
        """Context manager to avoid pushing the selected panel configurations"""
        if self._selected_panel_configs_watchers is not None:
            for etype, config in self.selected_panel_configs.items():
                config.param.unwatch(self._selected_panel_configs_watchers[etype])
        try:
            yield
        finally:
            self._selected_panel_configs_watchers = self.watch_selected_panel_configs()

    def update_entity_list(self, *events):
        """Update the entity list"""
        state = self.state
        for etype, selected in self.selected_entities.items():
            selected.param.selection.objects = state.entity_idx(etype).tolist()

    def pull_selected_configs(self, *events):
        """Pull the selected configurations"""
        state = self.state
        config_dict = {
            etype.to_state_type(): [config]
            for etype, config in self.selected_configs.items()
        }
        with self.dont_push_selected_configs():
            # Todo: check if for loop below is still required
            for etype, selected in self.selected_entities.items():
                config_dict[etype.to_state_type()][0].idx = int(
                    state.ent_idx(etype.to_state_type(), selected.selection[0])
                )
            converters.set_configs_from_state(state, config_dict)
        return state

    def pull_selected_panel_configs(self, *events):
        """Pull the selected panel configurations"""
        with self.dont_push_selected_panel_configs():
            for etype, panel_config in self.selected_panel_configs.items():
                panel_config.param.update(
                    **self.panel_configs[etype.to_state_type()][
                        self.selected_entities[etype].selection[0]
                    ].to_dict()
                )

    def pull_all_data(self):
        """Pull all the data from the simulator"""
        self.pull_selected_configs()
        self.pull_configs({StateType.SIMULATOR: self.configs[StateType.SIMULATOR]})

    def push_selected_to_config_list(self, *events):
        """Push the selected configurations to the configuration list"""
        lg.info("Push_selected_to_config_list %d", len(events))
        for e in events:
            if isinstance(e.obj, PanelConfig):
                stype = panel_config_to_stype[type(e.obj)]
            else:
                stype = config_to_stype[type(e.obj)]
            selected_entities = self.selected_entities[stype.to_entity_type()].selection
            for idx in selected_entities:
                if isinstance(e.obj, PanelConfig):
                    setattr(self.panel_configs[stype][idx], e.name, e.new)
                else:
                    setattr(self.configs[stype][idx], e.name, e.new)


if __name__ == "__main__":
    simulator = PanelController(client=SimulatorGRPCClient())
