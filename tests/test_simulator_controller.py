import jax.numpy as jnp

from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import AgentState
from vivarium.controllers.simulator_controller import SimulatorController, ControllerEntity
from vivarium.environments.braitenberg.selective_sensing.state import AgentState
from vivarium.utils.scene_configs import SceneConfiguration

NUM_STEPS = 10

def test_base_entity():
    config = SceneConfiguration('braitenberg')
    state = config.create_state()
    entity = ControllerEntity(state, 0, config.entity_types[0])
    
    entity.x_position = 10
    entity.apply_to_state(state)


def test_simulator_controller():
    simulator = SceneConfiguration('braitenberg').create_simulator()
    controller = SimulatorController(simulator)
    controller.step()

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag = controller.agents[idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.behavior = [3, 2, 1, 5]
    controller.apply_changes()
    controller.update_state()

    assert jnp.equal(jnp.array([3, 2, 1, 5]), controller.state.field(AgentState).behavior[idx]).all()

    for ag in controller.agents:
        ag.behavior = 5
        ag.motor = [0., 0.]

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.step()
        assert (not jnp.equal(pos, ag.position_center).all())

    controller.simulator_parameters.freq = -10
    controller.simulator_parameters.box_size = 42.
    controller.apply_changes()

    assert controller.client.freq == -10
    assert controller.client.box_size == 42.
    assert controller.client.env.box_size == 42.
