import jax.numpy as jnp

from vivarium.controllers.notebook_controller import NotebookController

NUM_STEPS = 4


def test_notebook_controller(scene_config):
    """Test default simulator run"""
    config = scene_config('braitenberg')
    controllers = config.create_controllers()
    simulator = config.create_simulator()
    controller = NotebookController(client=simulator, subtypes=config.config.subtypes, **controllers)
    agent_field = 'agents'
    controller.step()

    assert controller.client.freq == -1

    def beh(agent):
        left, right = agent.sensors()
        return 1 - right, 1 - left

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag =controller.agents[idx]

    ag.attach_behavior(beh)
    
    controller.execute_routines_and_behaviors()

    assert (jnp.equal(pos, ag.position_center).all())

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.run(threaded=False, num_steps=1)
        assert (not jnp.equal(pos, ag.position_center).all())

    ag.behavior = [3, 1, 2, 0]
    controller.step()
    assert jnp.equal(jnp.array([3, 1, 2, 0]), getattr(controller.state, agent_field).behavior[idx]).all()

    ag.color = 'pink'
    controller.step()
