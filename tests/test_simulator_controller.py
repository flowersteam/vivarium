import jax.numpy as jnp
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    init_state,
    SelectiveSensorsEnv,
)
from vivarium.simulator.simulator import Simulator

from vivarium.controllers.simulator_controller import SimulatorController

NUM_STEPS = 10


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

    for ag in controller.agents:
        ag.behavior = 5
        ag.motor = [0., 0.]
    # ag.motor = [1., 0.7]
    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.step()
        assert (not jnp.equal(pos, ag.position_center).all())

    # assert (not jnp.equal(controller.state.entity_state.position_center[idx], pos).all())

    # for _ in range(NUM_STEPS):
    #     simulator.step()

    # assert simulator
