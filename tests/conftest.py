"""
Test Fixtures for Vivarium

Server Fixtures Guide:
-----------------------
This module provides different server fixtures for different testing needs:

1. `grpc_client` (unit tests - fast)
   - Creates an in-process gRPC server and client (no subprocess)
   - Use for unit tests of controller logic
   - Fast (~ms startup), no process overhead
   - Example: testing controller state changes, parameter sync

2. `server_fixture` (integration tests - subprocess)
   - Creates a subprocess server via `start_simulation_server()`
   - Use for integration tests that need real subprocess behavior
   - Slower (~1-2s startup), full process isolation
   - Depends on `clean_server_state` for cleanup
   - Example: testing reconnection logic, server lifecycle

3. `server_and_interface_fixture` (full stack tests)
   - Creates subprocess server + Panel interface
   - Use for end-to-end tests including the web interface
   - Slowest, full stack
   - Depends on `clean_server_state` for cleanup

4. `VivariumController.start_server_process()` (controller-managed server)
   - Not a fixture - controller manages its own subprocess server
   - Use when testing VivariumController's server management capability
   - Example: testing `start_server=True` constructor parameter

Cleanup Fixtures:
-----------------
- `clean_server_state`: Kills any running server processes before a test.
  Required when using subprocess fixtures (`server_fixture`, `server_and_interface_fixture`)
  or when directly calling `start_simulation_server()`.

- `cleanup_vivarium_processes_session`: Session-scoped fixture that cleans up
  before and after the entire test session.
"""

import pytest
import time

from vivarium.interface.utils import cleanup_parameterized_class
from vivarium.utils.scene_configs import load_config, component_factories_from_config
from vivarium.utils.handle_server_interface import (
    wait_for_grpc_server,
    kill_all_vivarium_processes,
    start_simulation_server,
    stop_simulation_server,
    start_panel_interface,
    stop_panel_interface,
)
from vivarium.simulator.grpc_server.simulator_server import create_grpc_server
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.environment import Environment
from vivarium.environment.state import create_state_cls
from vivarium.interface.panel_app import create_interfaces
from vivarium.controllers import VivariumController
from vivarium.simulator import Simulator


@pytest.fixture(scope="session", autouse=True)
def cleanup_vivarium_processes_session():
    """Clean up any leftover Vivarium processes before and after the test session."""
    kill_all_vivarium_processes(include_clients=False)
    yield
    kill_all_vivarium_processes(include_clients=False)


@pytest.fixture
def clean_server_state():
    """Ensure no server is running before a test. Use for tests that spawn subprocess servers."""
    kill_all_vivarium_processes(include_clients=False)
    yield


@pytest.fixture
def server_fixture(clean_server_state):
    """Subprocess server fixture for integration tests.

    Use this fixture when you need a real subprocess server (not in-process).
    Automatically depends on `clean_server_state` for cleanup.

    Returns a namespace with `start` and `stop` methods for fine-grained control.
    All started servers are tracked and cleaned up automatically at teardown.

    Example (simple):
        def test_my_integration_test(server_fixture):
            server_process = server_fixture.start('braitenberg')
            # server is now running, test your integration scenario
            # cleanup is automatic

    Example (multiple servers):
        def test_reconnection(server_fixture):
            server1 = server_fixture.start('braitenberg')
            # ... use server1 ...
            server_fixture.stop(server1)
            server2 = server_fixture.start('braitenberg')
            # ... use server2 ...
            # cleanup of any remaining servers is automatic
    """
    servers = []

    class ServerManager:
        def start(self, scene_name, timeout=30.0):
            server_process = start_simulation_server(scene_name, timeout=timeout)
            time.sleep(1)
            servers.append(server_process)
            return server_process

        def stop(self, server_process):
            if server_process in servers:
                stop_simulation_server(server_process)
                servers.remove(server_process)

    yield ServerManager()

    # Cleanup any remaining servers
    for server in servers:
        stop_simulation_server(server)


@pytest.fixture
def server_and_interface_fixture(clean_server_state):
    """Subprocess server + Panel interface fixture for full-stack tests.

    Use this fixture when you need both the server and web interface running.
    Automatically depends on `clean_server_state` for cleanup.

    Example:
        def test_my_fullstack_test(server_and_interface_fixture):
            server_process, interface_process = server_and_interface_fixture('braitenberg')
            # both server and interface are now running
            # cleanup is automatic
    """
    server_process = None
    interface_process = None

    def _start(scene_name, timeout=30.0):
        nonlocal server_process, interface_process
        server_process = start_simulation_server(scene_name, timeout=timeout)
        interface_process, _ = start_panel_interface(show_output=False)
        time.sleep(1)
        return server_process, interface_process

    yield _start

    # Cleanup
    if interface_process is not None:
        stop_panel_interface(interface_process)
    if server_process is not None:
        stop_simulation_server(server_process)


@pytest.fixture(autouse=True)
def cleanup_parameterized_class_fixture(request):
    """
    Remove the dynamically added parameters from the Param classes
    as they might be remnants from previous tests
    """
    cleanup_parameterized_class()


@pytest.fixture
def scene_config():
    """Factory: load a Hydra scene config by name."""
    def fn(scene_name, overrides=[]):
        return load_config('scene', scene_name, overrides=overrides)
    return fn


@pytest.fixture
def environment_from_config(scene_config):
    """Factory: create an Environment from a scene name."""
    def fn(scene_name):
        return Environment.from_config(scene_config(scene_name).environment)
    return fn


@pytest.fixture
def state_from_config(scene_config):
    """Factory: create a state class from a scene name."""
    def fn(scene_name):
        config = scene_config(scene_name)
        base_state_cls = config.environment.base_state_cls
        update_fns = [f.update_state_cls for f in component_factories_from_config(config.environment.components)]
        return create_state_cls(base_state_cls, update_fns)
    return fn


@pytest.fixture
def simulator_from_config(scene_config):
    """Factory: create an in-process Simulator from a scene name."""
    def fn(scene_name, overrides=[]):
        return Simulator.from_config(scene_config(scene_name, overrides=overrides).simulator)
    return fn


@pytest.fixture
def vivarium_controller():
    """Factory: wrap a client (Simulator or gRPC) in a VivariumController."""
    def fn(client):
        return VivariumController(client=client, start_controller_thread=False)
    return fn


@pytest.fixture
def vivarium_controller_from_config(simulator_from_config):
    """Factory: create a VivariumController with in-process Simulator from a scene name."""
    def fn(scene_name, overrides=[]):
        client = simulator_from_config(scene_name, overrides=overrides)
        return VivariumController(client=client, start_controller_thread=False)
    return fn


@pytest.fixture
def vivarium_controller_start_session(grpc_client):
    """Factory: create a VivariumController via start_session with gRPC client. Closes on teardown."""
    controllers = []
    def fn(scene_name, overrides=[]):
        client = grpc_client(scene_name, overrides)
        controller = VivariumController.start_session(
            scene_name=scene_name,
            client=client,
            start_interface=False,
            start_controller_thread=False
        )
        controllers.append(controller)
        return controller
    
    yield fn
    
    for controller in controllers:
        controller.close()


@pytest.fixture
def controller_and_interfaces_from_config(scene_config, vivarium_controller):
    """Factory: create a VivariumController and corresponding Interface instances from a client."""
    def fn(client):
        controller = vivarium_controller(client)
        config = scene_config(controller.client.scene_name)
        interfaces = create_interfaces(
            config.environment.components.component_list,
            controller.controllers,
            controller.client.state,
        )
        return controller, interfaces
    return fn


@pytest.fixture
def grpc_server(simulator_from_config):
    """Factory: start an in-process gRPC server and return its address. Stops on teardown."""
    servers = []

    def fn(scene_name, overrides=[]):
        simulator = simulator_from_config(scene_name, overrides=overrides)
        server, port = create_grpc_server(simulator, port=50051)
        servers.append(server)
        
        # Wait for server to be ready using health check
        if not wait_for_grpc_server(port=port, timeout=10):
            raise RuntimeError(f"Test gRPC server did not start within 10 seconds")
        
        return f'localhost:{port}'
       
    yield fn
    
    for server in servers:
        server.stop(grace=5)


@pytest.fixture
def grpc_client(grpc_server):
    """Factory: create a SimulatorGRPCClient connected to an in-process server. Closes on teardown."""
    clients = []
    def fn(scene_name, overrides=[]):
        client = SimulatorGRPCClient(server=grpc_server(scene_name, overrides))
        clients.append(client)
        return client
    
    yield fn
    
    for client in clients:
        client.close()

