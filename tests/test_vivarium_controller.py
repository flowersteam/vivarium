import pytest
import jax.numpy as jnp

from vivarium.environment.components.entities.braitenberg.behaviors import Behaviors, behavior_params
from vivarium.controllers.vivarium_controller import VivariumController


NUM_STEPS = 10


@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_load_viviarium_controller(client_fixture, request):
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = VivariumController.from_client(client=client)
    controllers = controller.controllers
    
    controller.simulator_step()
    
    assert hasattr(controller.client.controller_parameters.agents, 'behaviors')

    idx = 0
    pos = controller.client.state.entity_state.position[idx]

    ag = controllers['agents'][idx]
    assert (jnp.equal(pos, ag.position_center).all())

    ag.behaviors[1].label = Behaviors.LOVE
    controller.apply_changes()
    
    assert ag.behaviors[1].label == Behaviors.LOVE

    assert jnp.equal(
        controller.client.state.agents.behavior_params[idx, 1],
        behavior_params[Behaviors.LOVE]
    ).all()

    for ag in controllers['agents']:
        ag.behaviors[0] = Behaviors.FEAR
            # ag.motor = [0., 0.]

    controller.apply_changes()
    
    assert jnp.equal(
        controller.client.state.agents.behavior_params[:, 0, :, :],
        jnp.full_like(controller.client.state.agents.behavior_params[:, 0, :, :], behavior_params[Behaviors.FEAR])
    ).all()

    for _ in range(NUM_STEPS):
        pos = controller.client.state.entity_state.position[idx]
        controller.simulator_step()
        assert (not jnp.equal(pos, ag.position).all())

    for ag in controllers['agents']:
        ag.behaviors[0] = Behaviors.MANUAL
        ag.motor = [1., 0.]

    controller.simulator_step()

    controller.apply_changes()

    controllers['collision'].epsilon = 42.
    controllers['collision'].alpha = 43.
    controller.apply_changes()
    assert controller.client.state.collision_state.epsilon.item() == 42.
    assert controller.client.state.collision_state.alpha.item() == 43.
    assert controller.client.state.collision_state.alpha.item() == 43.

    controllers['simulator'].freq = -10
    controllers['simulator'].env.box_size = 41.
    controllers['simulator'].env.box_size = 42.
    controllers['simulator'].env.num_scan_steps = 42
    controller.apply_changes()

    assert controllers['simulator'].freq == -10
    assert controllers['simulator'].env.box_size == 42.
    assert controllers['simulator'].env.num_scan_steps == 42
    assert controller.client.controller_parameters.simulator.freq == -10
    assert controller.client.controller_parameters.simulator.env.box_size == 42.
    assert controller.client.controller_parameters.simulator.env.num_scan_steps == 42

    controllers['agents'][0].color = 'pink'
    controllers['objects'][2].visible = False
    controller.apply_changes()
    assert controller.client.controller_parameters.agents.color[0] == 'pink'
    assert not controller.client.controller_parameters.objects.visible[2]
    

@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_controller_parameter_sync(client_fixture, request):
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller_1 = VivariumController.from_client(client=client)
    controller_2 = VivariumController.from_client(client=client)

    agents_1 = controller_1.controllers['agents']
    agents_2 = controller_2.controllers['agents']
    
    assert agents_1[0].color != 'pink' and agents_2[0].color != 'pink'
    
    # Empty potential changes from initialization
    controller_1.fetch_changes()
    controller_2.fetch_changes()

    agents_1[0].color = 'pink'
    
    controller_1.apply_changes()
    controller_2.apply_changes()
    
    assert agents_1[0].color == agents_2[0].color and agents_2[0].color == 'pink'
    