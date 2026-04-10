"""Tests for server and interface start/stop utilities.

These tests use subprocess server fixtures from conftest.py to verify
that the start/stop utilities work correctly. They also serve as smoke
tests to ensure the server and interface can start and respond to requests.
"""

import pytest

from vivarium.runtime import check_server_running, wait_for_http

pytestmark = pytest.mark.slow


def test_start_stop_server(server_fixture):
    """Test starting and stopping just the server.

    Verifies:
    - Server process starts successfully
    - Server responds to gRPC health checks
    """
    scene_name = "braitenberg"
    server_process = server_fixture.start(scene_name, timeout=30.0)
    assert server_process is not None
    assert server_process.poll() is None  # Process is still running
    # Verify server responds to health checks
    assert check_server_running(), "Server not responding to gRPC health check"


def test_start_stop_server_and_interface(server_and_interface_fixture):
    """Test starting and stopping server with interface.

    Verifies:
    - Server process starts successfully
    - Interface process starts successfully
    - Server responds to gRPC health checks
    - Interface responds to HTTP requests
    """
    scene_name = "braitenberg"
    server_process, interface_process = server_and_interface_fixture(scene_name, timeout=60.0)
    assert server_process is not None
    assert interface_process is not None
    assert server_process.poll() is None  # Process is still running
    assert interface_process.poll() is None  # Process is still running
    # Verify server responds to health checks
    assert check_server_running(), "Server not responding to gRPC health check"
    # Verify interface responds to HTTP requests
    assert wait_for_http(
        'http://localhost:5006/run_interface',
        retries=10,
        delay=2.0
    ), "Interface not responding to HTTP requests"
