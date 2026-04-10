"""
Jupyter notebook server lifecycle management.

Start, stop, and check Jupyter servers. Manages a registry of ports
started by the current interface instance for cleanup tracking.
"""

import os
import subprocess
import time
import logging

from vivarium.runtime.paths import (
    get_app_root, get_jupyter_config_path, get_jupyter_command,
    DEFAULT_JUPYTER_PORT,
)


lg = logging.getLogger(__name__)

# Registry of Jupyter ports started by this interface instance
_started_jupyter_ports: set[int] = set()


def register_jupyter_port(port: int) -> None:
    """Register a Jupyter port as started by this interface instance."""
    _started_jupyter_ports.add(port)


def unregister_jupyter_port(port: int) -> None:
    """Unregister a Jupyter port (e.g., when manually stopped)."""
    _started_jupyter_ports.discard(port)


def get_started_jupyter_ports() -> set[int]:
    """Get all Jupyter ports started by this interface instance."""
    return _started_jupyter_ports.copy()


def find_next_available_port(start_port: int = DEFAULT_JUPYTER_PORT, max_attempts: int = 100) -> int:
    """Find the next available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        The first available port found

    Raises:
        RuntimeError: If no available port found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if not check_jupyter_running(port):
            return port
    raise RuntimeError(f"No available port found between {start_port} and {start_port + max_attempts}")


def start_jupyter_server(port=DEFAULT_JUPYTER_PORT, notebook_dir=None, show_output=True, return_process_object=True):
    """Start a Jupyter notebook server with iframe-friendly configuration.

    In development mode: uses 'jupyter notebook' CLI command
    In frozen mode: spawns vivarium-jupyter executable

    :param port: Port to run Jupyter on, defaults to DEFAULT_JUPYTER_PORT
    :param notebook_dir: Directory to start Jupyter in, defaults to project root
    :param show_output: Whether to show Jupyter server output
    :param return_process_object: Deprecated parameter (kept for backward compatibility), always returns Popen object
    :return: Popen process object
    :raises RuntimeError: If the requested port is already in use
    """
    # Check if the requested port is already in use
    if check_jupyter_running(port):
        raise RuntimeError(
            f"Port {port} is already in use. Please stop the existing Jupyter server or choose a different port."
        )

    config_path = get_jupyter_config_path()
    if notebook_dir is None:
        # In frozen mode, notebooks are at distribution root (not in _internal)
        # In dev mode, get_app_root() returns the same as get_bundle_root()
        notebook_dir = get_app_root()

    jupyter_command = get_jupyter_command(
        port=port,
        notebook_dir=notebook_dir,
        config_path=config_path
    )

    lg.info(f"Starting Jupyter notebook server on port {port}...")
    lg.info(f"Notebook directory: {notebook_dir}")
    lg.info(f"Command: {' '.join(jupyter_command)}")

    # Set environment variable so notebooks can detect they were launched from Panel
    env = os.environ.copy()
    env['VIVARIUM_JUPYTER_FROM_PANEL'] = '1'

    jupyter_process = subprocess.Popen(
        jupyter_command,
        stdout=None if show_output else subprocess.DEVNULL,
        stderr=None if show_output else subprocess.DEVNULL,
        env=env
    )
    lg.info(f"Jupyter server started (PID: {jupyter_process.pid})")

    lg.info(f"Access it at: http://localhost:{port}")

    return jupyter_process


def check_jupyter_running(port=DEFAULT_JUPYTER_PORT):
    """Check if a Jupyter server is running on the specified port

    :param port: Port to check
    :return: True if Jupyter is running, False otherwise
    """
    import socket
    # Try both 127.0.0.1 and localhost to handle IPv4/IPv6 differences
    for host in ('127.0.0.1', 'localhost'):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                if result == 0:
                    return True
        except Exception:
            pass
    return False


def stop_jupyter_server(jupyter_process=None, port=DEFAULT_JUPYTER_PORT):
    """Stop a Jupyter server process

    :param jupyter_process: Process object to terminate (if available)
    :param port: Port to find and kill Jupyter on (fallback if process object not available)
    """
    from vivarium.runtime._process import kill_port_processes

    killed = False

    # Try to kill the process object if provided (use SIGKILL for Jupyter)
    if jupyter_process:
        try:
            if hasattr(jupyter_process, 'kill'):
                jupyter_process.kill()  # Use kill() directly, not terminate()
                if hasattr(jupyter_process, 'wait'):
                    jupyter_process.wait(timeout=3)
                killed = True
                lg.info("Jupyter server killed via process object")
            elif hasattr(jupyter_process, 'terminate'):
                # For subprocess.Popen
                jupyter_process.terminate()
                try:
                    jupyter_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    jupyter_process.kill()
                killed = True
                lg.info("Jupyter server terminated via subprocess.Popen")
        except Exception as e:
            lg.warning(f"Failed to kill Jupyter via process object: {e}")

    # Always check port and kill any remaining process (Jupyter can fork)
    lg.info(f"Checking if Jupyter is still running on port {port}...")
    if check_jupyter_running(port):
        lg.warning(f"Jupyter still detected on port {port}, force killing...")
        killed_pids = kill_port_processes(port, servers_only=True)
        if killed_pids:
            killed = True
            lg.info(f"Killed Jupyter processes: {killed_pids}")
    else:
        lg.info(f"No Jupyter process found on port {port}")

    if killed:
        # Give it a moment to clean up
        time.sleep(0.5)
