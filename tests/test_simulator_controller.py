import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
    EntityType
)
from vivarium.simulator.simulator import Simulator
from vivarium.controllers.simulator_controller import SimulatorController, ControllerEntity

NUM_STEPS = 10

def test_base_entity():
    state = init_state()
    entity = ControllerEntity(state, 0, EntityType.AGENT)
    
    entity.x_position = 10
    entity.apply_to_state(state)


def test_simulator_controller():
    """Test default simulator run"""
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)
    controller = SimulatorController(simulator)
    controller.step()

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag =controller.agents[idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.behavior = [3, 2]
    controller.apply_changes()
    controller.update_state()

    assert jnp.equal(jnp.array([3, 2]), controller.state.agent_state.behavior[idx]).all()

    for ag in controller.agents:
        ag.behavior = 5
        ag.motor = [0., 0.]

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.step()
        assert (not jnp.equal(pos, ag.position_center).all())

    controller.simulator_state.freq = -10
    controller.apply_changes()

    assert controller.state.simulator_state.freq != -10
    controller.update_state()
    assert controller.state.simulator_state.freq == -10

