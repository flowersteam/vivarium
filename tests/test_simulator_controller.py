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


def test_load_simulator_controller(simulator_from_config):
    simulator = simulator_from_config('braitenberg')
    controller = SimulatorController.from_client(client=simulator)
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

    for ag in controllers['agents']:
        ag.behaviors[0] = Behaviors.MANUAL
        ag.motor = [1., 0.]

    controller.step()

    controller.apply_changes()
    controller.update_state()

    controllers['collision'].epsilon = 42.
    controllers['collision'].alpha = 43.
    controller.apply_changes()
    controller.update_state()
    assert controller.state.collision_eps.item() == 42.
    assert controller.state.collision_alpha.item() == 43.

    controllers['simulator'].freq = -10
    controllers['simulator'].env.box_size = 41.
    controllers['simulator'].env.box_size = 42.
    controllers['simulator'].env.num_scan_steps = 42
    controller.apply_changes()

    assert controllers['simulator'].freq == -10
    assert controllers['simulator'].env.box_size == 42.
    assert controllers['simulator'].env.num_scan_steps == 42
    assert controller.client.freq == -10
    assert controller.client.env.box_size == 42.
    assert controller.client.env.num_scan_steps == 42

    controllers['agents'][0].color = 'pink'
    controllers['objects'][2].visible = False
    controller.apply_changes()
    assert controller.client.controller_parameters.agents.color[0] == 'pink'
    assert not controller.client.controller_parameters.objects.visible[2]
    
    
def test_controller_parameter_sync(simulator_from_config):
    simulator = simulator_from_config('braitenberg')
    controller_1 = SimulatorController.from_client(client=simulator)
    controller_2 = SimulatorController.from_client(client=simulator)

    agents_1 = controller_1.controllers['agents']
    agents_2 = controller_2.controllers['agents']
    
    # Empty potential changes from initialization
    controller_1.fetch_changes()
    controller_2.fetch_changes()

    agents_1[0].color = 'pink'

    assert agents_1[0].color != agents_2[0].color
    
    controller_1.apply_changes()
    controller_2.apply_changes()
    
    assert agents_1[0].color == agents_2[0].color and agents_2[0].color == 'pink'
    