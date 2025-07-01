import jax.numpy as jnp

from vivarium.environments.entities.braitenberg.controller import BraitenbergController

NUM_STEPS = 10


def test_base_entity(environment_and_state, braitenberg):
    env, state = environment_and_state(braitenberg)
    entity_type = 'agents'
    braitenberg_controller = BraitenbergController(entity_type, color=('red',) * state.agents.count())
    idx = 0
    entity = braitenberg_controller.controller(state)[idx]
    
    entity.x_position = 10
    state = entity.apply_to_state(state)
    assert state.entity_state.position[0, 0] == 10

    entity.color = 'pink'

    changes = entity._controller_change_recorder.fetch_changes()
    assert changes['color'][0]['__value'] == 'pink'


def test_simulator_controller(simulator_controller_from_config):
    controller = simulator_controller_from_config('braitenberg')
    controller.step()

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag = controller.agents[idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.behavior = [3, 2, 1, 5]
    controller.apply_changes()
    controller.update_state()

    assert jnp.equal(jnp.array([3, 2, 1, 5]), controller.state.agents.behavior[idx]).all()

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

    controller.agents[0].color = 'pink'
    controller.objects[2].visible = False
    controller.apply_changes()
    assert controller.client.controller_parameters.agents.color[0] == 'pink'
    assert not controller.client.controller_parameters.objects.visible[2]
