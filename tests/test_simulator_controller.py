import jax.numpy as jnp

from vivarium.environment.components.entities.braitenberg.behaviors import Behaviors, behavior_params
from vivarium.environment.components.entities.braitenberg.controller import BraitenbergController
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator import Simulator

NUM_STEPS = 10


def test_base_entity(environment_and_state, braitenberg):
    env, state = environment_and_state(braitenberg)
    entity_type = 'agents'
    braitenberg_controller = BraitenbergController(entity_type, state, 
                                                   color=['red'] * state.agents.count(),
                                                   visible=[True] * state.agents.count(),
                                                   hide_non_existing=[True] * state.agents.count()
                                                   )
    idx = 0
    entity = braitenberg_controller[idx]
    
    entity.x_position = 10
    state = entity.apply_to_state(state)
    assert state.entity_state.position[0, 0] == 10

    entity.color = 'pink'

    changes = entity._controller_change_recorder.fetch_changes()
    assert changes['color'][0]['__value'] == 'pink'


def test_load_simulator_controller(scene_config):
    config = scene_config('braitenberg')
    component_config = config.environment.components
    simulator = Simulator.from_config(config.simulator)
    controller = SimulatorController.from_config(component_config, client=simulator)
    controllers = controller.controllers
    controller.step()

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag = controllers['agents'][idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.behaviors[1].label = Behaviors.LOVE
    controller.apply_changes()
    controller.update_state()
    
    assert ag.behaviors[1].label == Behaviors.LOVE

    assert jnp.equal(
        controller.state.agents.behavior_params[idx, 1],
        behavior_params[Behaviors.LOVE]
    ).all()

    for ag in controllers['agents']:
        ag.behaviors[0] = Behaviors.FEAR
            # ag.motor = [0., 0.]

    controller.apply_changes()
    controller.update_state()
    
    assert jnp.equal(
        controller.state.agents.behavior_params[:, 0, :, :],
        jnp.full_like(controller.state.agents.behavior_params[:, 0, :, :], behavior_params[Behaviors.FEAR])
    ).all()

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.step()
        assert (not jnp.equal(pos, ag.position_center).all())


    controllers['collision'].epsilon = 42.
    controllers['collision'].alpha = 43.
    controller.apply_changes()
    controller.update_state()
    assert controller.state.collision_eps.item() == 42.
    assert controller.state.collision_alpha.item() == 43.

    controller.simulator_parameters.freq = -10
    controller.simulator_parameters.box_size = 42.
    controller.apply_changes()

    assert controller.client.freq == -10
    assert controller.client.box_size == 42.
    assert controller.client.env.box_size == 42.

    controllers['agents'][0].color = 'pink'
    controllers['objects'][2].visible = False
    controller.apply_changes()
    assert controller.client.controller_parameters.agents.color[0] == 'pink'
    assert not controller.client.controller_parameters.objects.visible[2]