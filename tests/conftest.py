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

import grpc
import pytest
import time
from time import sleep
import jax.numpy as jnp
from concurrent import futures


from vivarium.environment.components.eco_evo import (
    ReproductionComponent, ConsumptionComponent, SpawnComponent, EnergyComponent
)
from vivarium.environment.components.entities.braitenberg.component import BraitenbergComponent
from vivarium.environment.components.proximity_map.component import ProximityMapComponent
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
from vivarium.simulator.grpc_server.simulator_server import SimulatorServerServicer, create_grpc_server
from vivarium.environment.components.physics.step.component import StepComponent
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.environment import Environment, NeighborManager, MaskFunction
from vivarium.environment.state import BaseState, create_state_cls
from vivarium.simulator.grpc_server import simulator_pb2_grpc
from vivarium.interface.panel_app import create_interfaces
from vivarium.controllers import VivariumController
from vivarium.environment import Environment
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
    def fn(scene_name, overrides=[]):
        return load_config('scene', scene_name, overrides=overrides)
    return fn


@pytest.fixture
def environment_from_config(scene_config):
    def fn(scene_name):
        return Environment.from_config(scene_config(scene_name).environment)
    return fn


@pytest.fixture
def state_from_config(scene_config):
    def fn(scene_name):
        config = scene_config(scene_name)
        base_state_cls = config.environment.base_state_cls
        update_fns = [f.update_state_cls for f in component_factories_from_config(config.environment.components)]
        return create_state_cls(base_state_cls, update_fns)
    return fn


@pytest.fixture
def simulator_from_config(scene_config):
    def fn(scene_name, overrides=[]):
        return Simulator.from_config(scene_config(scene_name, overrides=overrides).simulator)
    return fn


@pytest.fixture
def vivarium_controller():
    def fn(client):
        return VivariumController(client=client, start_controller_thread=False)
    return fn


@pytest.fixture
def vivarium_controller_from_config(simulator_from_config):
    def fn(scene_name, overrides=[]):
        client = simulator_from_config(scene_name, overrides=overrides)
        return VivariumController(client=client, start_controller_thread=False)
    return fn


@pytest.fixture
def vivarium_controller_start_session(grpc_client):
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


@pytest.fixture#(scope="module")
def grpc_server(simulator_from_config):
    
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
    clients = []
    def fn(scene_name, overrides=[]):
        client = SimulatorGRPCClient(server=grpc_server(scene_name, overrides))
        clients.append(client)
        return client
    
    yield fn
    
    for client in clients:
        client.close()

def factory_names(factories):
    return [f.name for f in factories]


def remove_duplicates(factories):
    no_duplicate = []
    for f in factories:
        if f.name not in factory_names(no_duplicate):
            no_duplicate.append(f)
    return no_duplicate


@pytest.fixture
def proximity_map(step, braitenberg):
    return [*step, *braitenberg, ProximityMapComponent('proximity_map', 0)]


@pytest.fixture
def spawn(braitenberg):
    spawn = SpawnComponent(
        name='spawn',
        precedence=1,
        default=dict(
            subtype=0,
            period=1,
            start=True,
            position_range=[50., 60., 50., 60.],
            orientation_range=[3., 3.2]
        )
    )
    return [*braitenberg, spawn]


@pytest.fixture
def consumption(proximity_map):
    consumption = ConsumptionComponent(
        name='consumption', 
        precedence=1, 
        test_consumption = dict(
            source_subtype=0, 
            target_subtype=1, 
            range=1.0,
            start=True
        ),
        consuming_in_entity_state=True
    )
    return [*proximity_map, consumption]


@pytest.fixture
def energy(consumption):
    energy_component = EnergyComponent(
        name='energy',
        precedence=2,
        entity_type='agents',
        energy_init=0.5,
        energy_max=1.,
        energy_decay=0.00001,
        energy_burst=0.7
    )
    return [*consumption, energy_component]


@pytest.fixture
def reproduction(energy):
    # Disable the ConsumptionComponent for the reproduction test by setting its range to 0.
    energy[-2].consumption_params_dict['test_consumption']['range'] = 0.
    reproduction = ReproductionComponent(
        name='reproduction', 
        precedence=3,
        entity_type='agents',
        subtype=-1,
        birth_energy_threshold=0.5,
        death_energy_threshold=0.1,
        birth_recovery_time=100,
        birth_radius=10.,
        birth_energy=0.5
    )
    return [*energy, reproduction]


@pytest.fixture
def braitenberg(step):
    n_agents = 4
    braitenberg = BraitenbergComponent(
        name='braitenberg',
        precedence=1,
        entity_type='agents',
        subtype=jnp.array([0, 0, 1, 1], dtype=int),
        position=jnp.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
        orientation=jnp.array([0., 0., 0., 0.]),
        mass=jnp.array([1., 1., 1., 1.]),
        diameter=jnp.full((n_agents, ), 4.),
        friction=jnp.full((n_agents, ), 1.),
        exists=jnp.full((n_agents,), 1, dtype=int),
        n_behaviors=4,
        n_subtypes=2,
        wheel_diameter=1.0,
        max_speed=1.0,
        proxs_dist_max=20.0,
        proxs_cos_min=0.)
    return [*step, braitenberg]


@pytest.fixture
def step():
    return [StepComponent('step', 10, 0.1, MaskFunction('exists'))]


@pytest.fixture
def environment():
    def fn(factories, debug_mode=False):
        nm = NeighborManager(box_size=100., neighbor_radius=150., dr_threshold=10.)
        env = Environment(
            neighbor_manager=nm,
            base_state_cls=BaseState,
            factories=remove_duplicates(factories),
            to_jit=not debug_mode,
            debug_mode=debug_mode
        )
        return env
    return fn


@pytest.fixture
def environment_and_state(environment):
    def fn(factories, debug_mode=False):
        env = environment(factories, debug_mode=debug_mode)
        state = env.init_state()
        return env, state
    return fn