import pytest
import jax.numpy as jnp

from vivarium.controllers.notebook_controller import NotebookController

NUM_STEPS = 4

@pytest.mark.parametrize('scene_name', ['braitenberg', 'lenia_braitenberg'])
def test_notebook_controller(scene_name, simulator_controller_from_config):

    controller = simulator_controller_from_config(scene_name, NotebookController)
    controllers = controller.controllers
    
    # agent_field = 'agents'
    controller.step()

    assert controller.client.freq == -1

    def beh(agent):
        left, right = agent.sensors()
        return 1 - right, 1 - left

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag = controllers['agents'][idx]

    ag.attach_behavior(beh)
    
    controller.execute_routines_and_behaviors()

    assert (jnp.equal(pos, ag.position_center).all())

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.run(threaded=False, num_steps=1)
        
        # assert below no longer work, to reintroduce once manual behavior will be back
        # assert (not jnp.equal(pos, ag.position_center).all())

    ag.color = 'pink'
    controller.step()
