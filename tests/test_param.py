import pytest

import jax.numpy as jnp


from vivarium.simulator.simulator import Simulator
from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.environments.state import to_rigid_body_state
from vivarium.environments.braitenberg import selective_sensing
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.controllers.panel_controller import Agent, Object, ParamSimulator
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import SelectiveSensorsEnv


scene_name = 'braitenberg'

def get_rigid_body_state():
    state = get_point_particle_state()
    state = state.set(entity_state=to_rigid_body_state(state.entity_state))
    return state

def get_point_particle_state():
    return SceneConfiguration(scene_name).create_state()

state = get_point_particle_state()
agent_field = state.field_name(selective_sensing.AgentState)
object_field = state.field_name(selective_sensing.ObjectState)



@pytest.mark.parametrize("idx, init_state_fn, entity_type", [
    # (2, get_rigid_body_state, EntityType.AGENT),
    # (3, get_rigid_body_state, EntityType.OBJECT),
    (4, get_point_particle_state, agent_field),
])
def test_param_entity(idx, init_state_fn, entity_type):
    state = init_state_fn()
    env = SelectiveSensorsEnv(state=state, box_size=100., neighbor_radius=100.)
    simulator = Simulator(env=env, scene_name=scene_name)
    controller = SimulatorController(simulator)

    controller_entities = getattr(controller, entity_type)
    
    entity_cls = Agent if entity_type == agent_field else Object
    entity = entity_cls(getattr(controller, entity_type), controller.subtype_labels)
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
    
    # TODO: Test for attributes that are not in the state (e.g. color, see also commented below)
    # entity.color = 'red'
    # controller.apply_changes()
    # controller.update_state()
    # assert (controller_entities[idx].color == string_to_rgb_array('red')).all()

    if entity_type == agent_field:
        entity.right_motor = 2.
        controller.apply_changes()
        controller.update_state()
        assert controller_entities[idx].motor[1] == 2.

    controller_entities[idx].position_orientation = 1.
    controller.apply_changes()
    controller.update_state()
    entity.update_from_server = True
    assert entity.orientation == 1.

    # controller_entities[idx].color = jnp.array([0.1, 0.2, 0.3])
    # controller.apply_changes()
    # controller.update_state()
    # entity.update_from_server = True
    # assert entity.color == rgb_array_to_string(jnp.array([0.1, 0.2, 0.3]))

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

    if entity_type == agent_field:
        controller_entities[idx].left_motor = 4.
        controller.apply_changes()
        controller.update_state()
        entity.update_from_server = True
        assert entity.left_motor == 4.


def test_simulator_state_param():
    state = get_rigid_body_state()
    env = SelectiveSensorsEnv(state=state, box_size=100., neighbor_radius=100.)
    simulator = Simulator(env=env, scene_name=scene_name)
    controller = SimulatorController(simulator)

    simulator_param = ParamSimulator(controller.client)
    simulator_param.update_from_server = True
    assert simulator_param.freq == controller.client.freq
    assert simulator_param.box_size == controller.client.box_size

    simulator_param.freq = -10

    assert controller.client.freq == -10
    assert controller.client.box_size == simulator_param.box_size
