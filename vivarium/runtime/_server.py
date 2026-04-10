"""
gRPC simulation server lifecycle management.

Start, stop, and health-check the Vivarium gRPC server.
"""

import os
import subprocess
import time
import logging

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from vivarium.runtime.paths import get_server_command
from vivarium.runtime._process import kill_port_processes


lg = logging.getLogger(__name__)


def wait_for_grpc_server(host="localhost", port=50051, timeout=30.0, poll_interval=0.5,
                         process=None, quiet=True):
    """
    Wait for the gRPC server to be ready using the standard health checking protocol.

    Args:
        host: Server hostname
        port: Server port
        timeout: Maximum seconds to wait
        poll_interval: Seconds between connection attempts
        process: Optional subprocess.Popen object to monitor. If the process
                 exits during waiting, the function returns early with failure.
        quiet: If True, don't log verbose error details on timeout (default: True)

    Returns:
        True if server is ready, False if timeout reached or process exited
    """
    start_time = time.time()
    attempt = 0
    last_error = None

    while time.time() - start_time < timeout:
        # Check if the server process has crashed
        if process is not None and process.poll() is not None:
            lg.error(f"Server process exited with code {process.returncode} during startup")
            return False

        attempt += 1
        # Create a fresh channel for each attempt to avoid cached connection states
        channel = grpc.insecure_channel(f"{host}:{port}")
        try:
            health_stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest(service="")
            response = health_stub.Check(request, timeout=2.0)
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                lg.debug(f"gRPC server ready after {attempt} attempts ({time.time() - start_time:.1f}s)")
                channel.close()
                return True
        except grpc.RpcError as e:
            last_error = e
        finally:
            channel.close()
        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    if quiet:
        lg.debug(f"gRPC server not ready after {attempt} attempts ({elapsed:.1f}s)")
    else:
        lg.warning(f"gRPC server not ready after {attempt} attempts ({elapsed:.1f}s). Last error: {last_error}")
    return False


def check_server_running(host="localhost", port=50051, timeout=1.0, poll_interval=0.2):
    """Check if the gRPC server is currently running.

    Args:
        host: Server hostname
        port: Server port
        timeout: Maximum seconds to wait for server to be ready
        poll_interval: Seconds between connection attempts
    Returns:
        True if server is running and responding to health checks, False otherwise
    """
    return wait_for_grpc_server(host=host, port=port, timeout=timeout, poll_interval=poll_interval)


def start_simulation_server(scene_name=None, timeout=30.0, show_output=False):
    """Start the simulation server for a given scene.

    Args:
        scene_name: Name of the scene configuration to load (e.g., 'session_1').
                   If None, uses Hydra's default configuration.
        timeout: Maximum seconds to wait for server to be ready
        show_output: Whether to show server output in console

    Returns:
        Popen process object for the server

    Raises:
        RuntimeError: If server doesn't start within timeout
    """

    if check_server_running():
        raise RuntimeError("A simulation server is already running")

    # Kill any zombie processes on port 50051 that aren't responding to health checks
    # This can happen if a previous server crashed without proper cleanup
    zombie_pids = kill_port_processes(50051, servers_only=True)
    if zombie_pids:
        lg.warning(f"Killed zombie process(es) on port 50051: {zombie_pids}")
        time.sleep(0.5)  # Give the OS time to release the port

    cmd_args = [f"scene={scene_name}"] if scene_name else []
    server_command = get_server_command(cmd_args)

    lg.info(f"Starting Vivarium server{f' with scene {scene_name!r}' if scene_name else ''}...")

    # Start in a new session/process group so it doesn't receive SIGINT when the
    # parent (interface) is interrupted with Ctrl-C. This allows clean shutdown.
    popen_kwargs = {
        'stdout': None if show_output else subprocess.DEVNULL,
        'stderr': None if show_output else subprocess.DEVNULL,
    }
    if os.name == 'nt':  # Windows
        # CREATE_NEW_PROCESS_GROUP prevents Ctrl-C from propagating to the subprocess
        popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:  # Unix (Linux, macOS)
        popen_kwargs['start_new_session'] = True

    server_process = subprocess.Popen(server_command, **popen_kwargs)

    # Wait for gRPC server to be ready, monitoring the process for crashes
    if not wait_for_grpc_server(timeout=timeout, process=server_process):
        # Check if the process crashed vs just not responding
        exit_code = server_process.poll()
        if exit_code is not None:
            raise RuntimeError(f"Server process crashed during startup (exit code: {exit_code})")

        # Process still running but not responding - terminate it
        server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
        raise RuntimeError(f"gRPC server did not start within {timeout} seconds")

    lg.info(f"Server started successfully (PID: {server_process.pid})")
    return server_process


def stop_simulation_server(server_process):
    """Stop a simulation server process.

    Args:
        server_process: Popen process object to terminate
    """
    if server_process is None:
        return

    lg.info(f"Stopping simulation server (PID: {server_process.pid})...")

    # Check if already dead
    if server_process.poll() is not None:
        lg.info(f"Server already exited (code: {server_process.returncode})")
        return

    try:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            lg.info("Server terminated gracefully")
            return
        except subprocess.TimeoutExpired:
            lg.warning("Server did not terminate gracefully, forcing kill...")
            server_process.kill()
            server_process.wait(timeout=3)
            lg.info("Server killed")
            return
    except Exception as e:
        lg.warning(f"Error stopping server via process handle: {e}")

    # Fallback: kill by port if process handle didn't work
    lg.info("Attempting fallback kill by port...")
    kill_port_processes(50051, servers_only=True)
