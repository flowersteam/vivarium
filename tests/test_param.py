from vivarium.utils.converters import rgb_array_to_string, string_to_rgb_array

import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state as init_rigid_body_state,
    SelectiveSensorsEnv,
    EntityType
)

from vivarium.controllers.panel_controller import Agent, Object, ParamSimulatorState
from vivarium.controllers.dataclass_wrapper import DataclassWrapper

from vivarium.environments.braitenberg import selective_sensing
from vivarium.environments.utils import rigid_body_to_point_particle

from vivarium.utils.scene_configs import load_scene_config
import pytest

init_state_point_particle, _ = rigid_body_to_point_particle(selective_sensing)


def get_rigid_body_state():
    config = load_scene_config('prey_predator')
    return init_rigid_body_state(**config)

def get_point_particle_state():
    config = load_scene_config('prey_predator')
    return init_state_point_particle(**config)

from vivarium.simulator.simulator import Simulator
from vivarium.controllers.simulator_controller import SimulatorController




@pytest.mark.parametrize("idx, init_state_fn, entity_type", [
    (2, get_rigid_body_state, EntityType.AGENT),
    (3, get_rigid_body_state, EntityType.OBJECT),
    (4, get_point_particle_state, EntityType.AGENT),
])
def test_param_entity(idx, init_state_fn, entity_type):
    state = init_state_fn()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)
    controller = SimulatorController(simulator)

    controller_entities = controller.agents if entity_type == EntityType.AGENT else controller.objects
    entity = Agent(controller.agents, controller.get_subtype_labels()) if entity_type == EntityType.AGENT else Object(controller.objects)
    entity.selection = [idx]
    entity.update_from_server = True

    assert entity.x_position == controller_entities[idx].position_center[0]
    entity.x_position = 10
    assert entity.x_position == 10
    controller.apply_changes()
    controller.update_state()
    assert controller_entities[idx].position_center[0] == 10
    entity.exists = False
    controller.apply_changes()
    controller.update_state()
    assert controller_entities[idx].exists.item() == 0
    entity.color = 'red'
    controller.apply_changes()
    controller.update_state()
    assert (controller_entities[idx].color == string_to_rgb_array('red')).all()

    if entity_type == EntityType.AGENT:
        entity.right_motor = 2.
        controller.apply_changes()
        controller.update_state()
        assert controller_entities[idx].motor[1] == 2.

    controller_entities[idx].position_orientation = 1.
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.orientation == 1.

    controller_entities[idx].color = jnp.array([0.1, 0.2, 0.3])
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.color == rgb_array_to_string(jnp.array([0.1, 0.2, 0.3]))

    controller_entities[idx].exists = jnp.array(1)
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.exists

    controller_entities[idx].x_position = 20
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.x_position == 20

    if entity_type == EntityType.AGENT:
        controller_entities[idx].left_motor = 4.
        controller.apply_changes()
        controller.update_state()
        entity.update_from_server = True
        assert entity.left_motor == 4.


def test_simulator_state_param():
    state = get_rigid_body_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)
    controller = SimulatorController(simulator)

    simulator_param = ParamSimulatorState(controller.simulator_state)
    simulator_param.update_from_server = True
    assert simulator_param.freq == controller.simulator_state.freq

    simulator_param.freq = -10

    controller.apply_changes()
    controller.update_state()
    # simulator_state = simulator_param.data.apply()

    assert controller.state.simulator_state.freq == -10
    assert simulator.state.simulator_state.box_size == controller.simulator_state.box_size