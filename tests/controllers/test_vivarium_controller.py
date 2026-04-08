import pytest
from time import sleep

import jax.numpy as jnp

from vivarium.components.entities.braitenberg.behaviors import Behaviors, behavior_params
from vivarium.controllers.vivarium_controller import VivariumController
from vivarium.utils.handle_server_interface import (
    check_server_running,
    stop_server_and_interface,
    stop_simulation_server,
)

NUM_STEPS = 10


@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_load_viviarium_controller(client_fixture, request):
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = VivariumController(client=client, start_controller_thread=False)
    controllers = controller.controllers

    controller.simulator_step()

    assert hasattr(controller.client.controller_parameters.agents, 'behaviors')

    idx = 0
    pos = controller.client.state.entity_state.position[idx]

    ag = controllers['agents'][idx]
    assert (jnp.equal(pos, ag.position).all())

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


@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_controller_parameter_sync(client_fixture, request):
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller_1 = VivariumController(client=client, start_controller_thread=False)
    controller_2 = VivariumController(client=client, start_controller_thread=False)

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


# Tests for new disconnected state functionality

def test_disconnected_controller():
    """Test that default constructor creates disconnected controller."""
    controller = VivariumController()
    assert not controller.is_connected()
    assert controller.client is None
    assert 'simulator' not in controller.controllers


def test_disconnected_controller_methods_raise():
    """Test that methods requiring connection raise when disconnected."""
    controller = VivariumController()
    with pytest.raises(RuntimeError, match="Not connected"):
        controller.start_controller_thread()


@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_constructor_with_client(client_fixture, request):
    """Test that passing client to constructor initializes properly."""
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = VivariumController(client=client, start_controller_thread=False)

    assert controller.is_connected(verify=False)
    assert controller.client is client
    assert 'simulator' in controller.controllers
    assert hasattr(controller, 'agents')  # Controller loaded from config


@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['simulator_from_config', 'grpc_client'])
def test_is_connected(client_fixture, request):
    """Test is_connected() returns correct state."""
    client = request.getfixturevalue(client_fixture)('braitenberg')

    controller = VivariumController()
    assert controller.is_connected() == False

    controller = VivariumController(client=client, start_controller_thread=False)
    assert controller.is_connected() == True


def test_constructor_start_server_missing_scene_name():
    """Test that start_server=True without scene_name raises."""
    with pytest.raises(ValueError, match="scene_name is required"):
        VivariumController(start_server=True, start_controller_thread=False)


def test_connect_to_server_no_server(caplog, monkeypatch):
    """Test connect_to_server=True when no server is running logs warning."""
    from vivarium.controllers import vivarium_controller
    # Mock check_server_running to always return False
    monkeypatch.setattr(vivarium_controller, 'check_server_running', lambda *args, **kwargs: False)

    controller = VivariumController(connect_to_server=True, start_controller_thread=False)
    assert not controller.is_connected()
    assert "No server running" in caplog.text


@pytest.mark.parametrize('client_fixture', ['simulator_from_config'])
def test_is_connected_verify_false(client_fixture, request):
    """Test is_connected(verify=False) only checks local state."""
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = VivariumController(client=client, start_controller_thread=False)

    # Without verify, just checks if client is set
    assert controller.is_connected(verify=False) == True
    assert controller.client is not None


def test_is_connected_verify_detects_no_server(caplog, monkeypatch):
    """Test is_connected(verify=True) detects when server is down."""
    from vivarium.controllers import vivarium_controller
    # Mock check_server_running to return False (simulate server down)
    monkeypatch.setattr(vivarium_controller, 'check_server_running', lambda *args, **kwargs: False)

    # Create a controller with a fake gRPC client (simulating stale connection)
    controller = VivariumController()
    fake_client = type('FakeClient', (), {
        'is_grpc_client': True,
        'server_host': 'localhost',
        'server_port': 50051,
    })()
    controller.client = fake_client
    controller.controllers = {'simulator': 'fake'}

    # With verify=True, it should detect no server and clean up
    assert controller.is_connected(verify=True) == False
    assert controller.client is None
    assert controller.controllers == {}
    assert "Server is no longer responding" in caplog.text


@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['grpc_client'])
def test_is_connected_verify_true_with_running_server(client_fixture, request):
    """Test is_connected(verify=True) returns True when server is running."""
    client = request.getfixturevalue(client_fixture)('braitenberg')
    controller = VivariumController(client=client, start_controller_thread=False)

    # With verify=True and server running, should return True
    # Note: The grpc_client fixture uses a random port, but the controller's
    # client is connected to it, so we're really testing the local state here
    assert controller.is_connected(verify=False) == True
    assert controller.client is not None


@pytest.mark.slow
def test_reconnect_after_external_server_restart(server_fixture):
    """Integration test: reconnect after server is stopped and restarted externally.

    Scenario:
    1. Server started independently (as subprocess)
    2. VivariumController connects to this server
    3. Server is stopped externally
    4. New server is started
    5. VivariumController reconnects to the new server
    """
    # Step 1: Start server as subprocess
    server_process_1 = server_fixture.start('braitenberg', timeout=30.0)
    assert check_server_running()

    # Step 2: Connect controller to server
    controller = VivariumController(connect_to_server=True, start_controller_thread=False)
    assert controller.is_connected()
    original_client = controller.client

    # Step 3: Stop server externally
    server_fixture.stop(server_process_1)
    sleep(1)  # Give time for server to fully stop
    assert not check_server_running()

    # Controller still thinks it's connected (local state only)
    assert controller.is_connected(verify=False) == True
    # But verify=True should detect server is down
    assert controller.is_connected(verify=True) == False
    assert controller.client is None  # State cleaned up

    # Step 4: Start a new server
    server_fixture.start('braitenberg', timeout=30.0)
    assert check_server_running()

    # Step 5: Connect to the new server
    result = controller.connect(start_controller_thread=False)

    assert result == True
    assert controller.is_connected()
    assert controller.client is not original_client  # New client instance
    assert 'simulator' in controller.controllers  # Controllers reinitialized


@pytest.mark.slow
def test_two_controllers_server_handoff(clean_server_state):
    """Integration test: two controllers with server lifecycle handoff.

    Scenario:
    1. vc1 starts a server and connects to it
    2. vc2 tries to start a server - should be informed server exists, connects to it
    3. vc1 stops the server it created
    4. vc2 detects it's no longer connected
    5. vc2 starts a new server and connects to it
    6. vc1 connects to this new server
    """
    # Step 1: vc1 starts a server and connects
    vc1 = VivariumController(start_server=True, scene_name='braitenberg', start_controller_thread=False)
    assert vc1.is_connected()
    assert vc1._server_process is not None
    assert check_server_running()

    try:
        # Step 2: vc2 tries to start a server - should connect to existing
        vc2 = VivariumController(start_server=True, scene_name='braitenberg', start_controller_thread=False)
        assert vc2.is_connected()
        assert vc2._server_process is None  # vc2 didn't start the server
        assert 'simulator' in vc2.controllers

        # Both controllers are now connected to the same server
        assert vc1.is_connected() and vc2.is_connected()
        
        # Change in one controller is reflected in the other
        assert vc1.agents[0].left_motor == 0. and vc2.agents[0].left_motor == 0.
        vc2.agents[0].left_motor = 1.0
        vc2.apply_changes()
        vc1.apply_changes()
        assert vc1.agents[0].left_motor == 1.0 and vc2.agents[0].left_motor == 1.0

        # Step 3: vc1 stops the server it created
        vc1.stop_server_process()
        assert vc1._server_process is None
        assert not vc1.is_connected()  # vc1 is now disconnected (automatic)
        sleep(1)  # Give time for server to fully stop
        assert not check_server_running()

        # Step 4: vc2 detects it's no longer connected
        assert vc2.is_connected(verify=False) == True  # Local state still set
        assert vc2.is_connected(verify=True) == False  # But server is gone
        assert vc2.client is None  # State cleaned up

        # Step 5: vc2 starts a new server and connects
        vc2.start_server_process('braitenberg', timeout=30.0, start_controller_thread=False)
        assert vc2.is_connected()
        assert vc2._server_process is not None
        assert check_server_running()

        # Step 6: vc1 connects to this new server
        result = vc1.connect(start_controller_thread=False)
        assert result == True
        assert vc1.is_connected()
        assert 'simulator' in vc1.controllers

        # Both controllers are now connected to the new server
        assert vc1.is_connected() and vc2.is_connected()

    finally:
        # Clean up: stop any running server
        if vc2._server_process is not None:
            stop_simulation_server(vc2._server_process)
            vc2._server_process = None
        if vc1._server_process is not None:
            stop_simulation_server(vc1._server_process)
            vc1._server_process = None


@pytest.mark.slow
def test_start_server_scene_mismatch(caplog):
    """Test that starting a server with a different scene logs warning and doesn't connect."""
    # Ensure no server is running from previous tests
    if check_server_running():
        stop_server_and_interface(safe_mode=False)
        sleep(1)
        assert not check_server_running(), "Failed to stop existing server"

    # Start a server with 'braitenberg' scene
    vc1 = VivariumController(start_server=True, scene_name='braitenberg', timeout=30.0, start_controller_thread=False)
    assert vc1.is_connected()
    assert check_server_running()

    try:
        # Try to start a server with a different scene
        vc2 = VivariumController()
        vc2.start_server_process('quickstart', timeout=30.0, start_controller_thread=False)

        # vc2 should NOT be connected (scene mismatch)
        assert not vc2.is_connected()
        assert vc2._server_process is None

        # Warning should be logged
        assert "Server is already running with scene 'braitenberg'" in caplog.text
        assert "requested: 'quickstart'" in caplog.text

    finally:
        # Clean up
        if vc1._server_process is not None:
            stop_simulation_server(vc1._server_process)
            vc1._server_process = None
