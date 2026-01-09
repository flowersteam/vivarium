import pytest
import jax.numpy as jnp


NUM_STEPS = 4


@pytest.mark.parametrize('scene_name', ['session_1', 'session_2', 'session_3', 'session_4'])
def test_session(scene_name, vivarium_controller_start_session):

    controller = vivarium_controller_start_session(scene_name, overrides=["environment.kwargs.debug_mode=true"])

    def beh(agent):
        left, right = agent.proximeters()
        return 1 - right, 1 - left

    idx = 0
    pos = controller.client.state.entity_state.position[idx]

    ag = controller.agents[idx]

    ag.attach_behavior(beh)
    
    assert (jnp.equal(pos, ag.position).all())
    
    controller.simulator.simulation_running = True
    
    # Step twice to initialize force and momentum
    controller.step()
    controller.step()

    for _ in range(NUM_STEPS):
        pos = controller.client.state.entity_state.position[idx]
        controller.step()
        assert (not jnp.equal(pos, ag.position).all())
    
    ag.color = 'pink'
    controller.step()
