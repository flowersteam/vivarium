import time

import pytest

from vivarium.utils.handle_server_interface import (
    start_simulation_server,
    stop_simulation_server,
    start_panel_interface,
    stop_panel_interface,
)


@pytest.fixture
def server_fixture(clean_server_state):
    """Fixture that starts a server and ensures cleanup after test."""
    server_process = None

    def _start_server(scene_name, timeout=30.0):
        nonlocal server_process
        server_process = start_simulation_server(scene_name, timeout=timeout)
        time.sleep(1)
        return server_process

    yield _start_server

    # Cleanup
    if server_process is not None:
        stop_simulation_server(server_process)


@pytest.fixture
def server_and_interface_fixture(clean_server_state):
    """Fixture that starts server and interface, ensures cleanup after test."""
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


def test_start_stop_server(server_fixture):
    """Test starting and stopping just the server."""
    scene_name = "braitenberg"
    server_process = server_fixture(scene_name, timeout=30.0)
    assert server_process is not None
    assert server_process.poll() is None  # Process is still running


def test_start_stop_server_and_interface(server_and_interface_fixture):
    """Test starting and stopping server with interface."""
    scene_name = "session_3"
    server_process, interface_process = server_and_interface_fixture(scene_name, timeout=60.0)
    assert server_process is not None
    assert interface_process is not None
    assert server_process.poll() is None  # Process is still running
    assert interface_process.poll() is None  # Process is still running
