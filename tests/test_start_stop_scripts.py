"""Tests for server and interface start/stop utilities.

These tests use subprocess server fixtures from conftest.py to verify
that the start/stop utilities work correctly.
"""


def test_start_stop_server(server_fixture):
    """Test starting and stopping just the server."""
    scene_name = "braitenberg"
    server_process = server_fixture.start(scene_name, timeout=30.0)
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
