import os
import time
import psutil
import subprocess
import signal
import logging
import re
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from vivarium.utils.runtime import get_app_root, get_jupyter_config_path, get_server_command, get_interface_command, get_jupyter_command


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


def find_next_available_port(start_port: int = 8889, max_attempts: int = 100) -> int:
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


SERVER_PROCESS_NAME = "scripts/run_server.py"
INTERFACE_PROCESS_NAME = "scripts/run_interface.py"
SERVER_PROCESS_NAME_WIN = "run_server.py"
INTERFACE_PROCESS_NAME_WIN = "run_interface.py"


def start_jupyter_server(port=8889, notebook_dir=None, show_output=True, return_process_object=True):
    """Start a Jupyter notebook server with iframe-friendly configuration.

    In development mode: uses 'jupyter notebook' CLI command
    In frozen mode: spawns vivarium-jupyter executable

    :param port: Port to run Jupyter on, defaults to 8889
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


def check_jupyter_running(port=8889):
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


def stop_jupyter_server(jupyter_process=None, port=8889):
    """Stop a Jupyter server process

    :param jupyter_process: Process object to terminate (if available)
    :param port: Port to find and kill Jupyter on (fallback if process object not available)
    """
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


def get_process_pids_unix(process_name: str):
    """Get the processes IDs of a running process by name

    :param process_name: process name
    :return: lisf of processes IDs
    """
    pids = []
    process = subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE)
    out, err = process.communicate()
    for line in out.splitlines():
        if process_name.encode("utf-8") in line:
            pid_str = line.split()[1]
            pid = pid_str.decode()
            lg.warning(
                f" Found the process {process_name} running with this PID: {pid}"
            )
            pids.append(pid)
    return pids


def get_process_pids_windows(process_name):
    """Get the processes IDs of a running process by name

    :param process_name: process name
    :return: list of processes IDs
    """
    pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"]).lower()
                if process_name.lower() in cmdline:
                    pid = proc.info["pid"]
                    lg.warning(
                        f" Found the process {process_name} running with this PID: {pid}"
                    )
                    pids.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


def get_server_interface_pids():
    """Get the process IDs of the server and interface

    :return: server and interface process IDs
    """
    if os.name == "nt":
        interface_pids = get_process_pids_windows(INTERFACE_PROCESS_NAME_WIN)
        server_pids = get_process_pids_windows(SERVER_PROCESS_NAME_WIN)
    elif os.name == "posix":
        interface_pids = get_process_pids_unix(INTERFACE_PROCESS_NAME)
        server_pids = get_process_pids_unix(SERVER_PROCESS_NAME)
    else:
        lg.error("OS not recognized")
        return

    return interface_pids, server_pids


def kill_process(pid):
    """Kill a process by its ID

    :param pid: process ID
    """
    os.kill(int(pid), signal.SIGTERM)
    lg.warning(f"Killed process with PID: {pid}")


def terminate_process(pids):
    """Terminate the process if the PID is not None"""
    if pids:
        for pid in pids:
            kill_process(pid)


def stop_server_and_interface(safe_mode=True):
    """Stop the server and interface"""
    processes_running = False

    interface_pids, server_pids = get_server_interface_pids()

    if interface_pids or server_pids:
        lg.info("\nStopping server and interface processes\n")
        processes_running = True
        if not safe_mode:
            terminate_process(interface_pids)
            terminate_process(server_pids)
            processes_running = False
        else:
            message = "\nThe following processes are running:\n"
            if interface_pids:
                message += f" - Interface (PIDs: {interface_pids})\n"
            if server_pids is not None:
                message += f" - Server (PIDs: {server_pids})\n"
            message += "Do you want to stop them? (y/n): "
            user_input = input(message)

            if user_input.lower() == "y":
                terminate_process(interface_pids)
                terminate_process(server_pids)
                processes_running = False

        if not processes_running:
            lg.info("\nServer and Interface processes have been stopped\n")

    return processes_running


def kill_port_processes(port, servers_only=True):
    """Kill processes on a specific port.

    Cross-platform: Uses lsof on Unix, netstat+taskkill on Windows.

    Args:
        port: Port number to clear
        servers_only: If True, only kill processes LISTENING on the port (servers).
                     If False, kill all processes on the port including clients.

    Returns:
        List of PIDs that were killed
    """
    killed_pids = []

    if os.name == 'nt':  # Windows
        try:
            result = subprocess.run(
                ['netstat', '-ano', '-p', 'TCP'],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f':{port}' in line:
                    if servers_only and 'LISTENING' not in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            try:
                                subprocess.run(
                                    ['taskkill', '/F', '/PID', pid],
                                    capture_output=True
                                )
                                killed_pids.append(pid)
                                lg.info(f"Killed process {pid} on port {port}")
                            except Exception:
                                pass
        except Exception as e:
            lg.warning(f"Failed to check port {port} on Windows: {e}")
    else:  # Unix (macOS, Linux)
        try:
            # Use lsof to find processes on the port
            if servers_only:
                # Only get servers (LISTEN state), not client connections
                cmd = ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"]
            else:
                # Get all processes on the port (servers and clients)
                cmd = ["lsof", "-ti", f":{port}"]

            result = subprocess.run(cmd, capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid and pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        killed_pids.append(pid)
                        lg.info(f"Killed process {pid} on port {port}")
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception as e:
            lg.warning(f"Failed to check port {port}: {e}")

    return killed_pids


def kill_vivarium_processes(server=False, clients=False, interface=False, jupyter=False,
                            grpc_port=50051, interface_port=5006, jupyter_port=8889,
                            only_tracked_jupyter=False):
    """Kill Vivarium-related processes selectively.

    Args:
        server: Kill the gRPC server
        clients: Kill gRPC clients (in addition to server)
        interface: Kill the Panel interface
        jupyter: Kill Jupyter server(s)
        grpc_port: Port for gRPC server
        interface_port: Port for Panel interface
        jupyter_port: Port for Jupyter (used when only_tracked_jupyter=False)
        only_tracked_jupyter: If True, only kill Jupyter servers from the tracked registry
                             (started by the interface instance). If False, kill the
                             specified jupyter_port.

    Returns:
        List of PIDs that were killed
    """
    killed = []
    if not (server or clients or interface or jupyter):
        lg.warning("No processes specified to kill.")
        return []
    if server:
        killed.extend(kill_port_processes(grpc_port, servers_only=not clients))
    if interface:
        killed.extend(kill_port_processes(interface_port, servers_only=False))
    if jupyter:
        if only_tracked_jupyter:
            # Only kill Jupyter servers we started (from registry)
            tracked_ports = get_started_jupyter_ports()
            for port in tracked_ports:
                killed.extend(kill_port_processes(port, servers_only=True))
        else:
            # Legacy behavior: kill specified jupyter_port
            killed.extend(kill_port_processes(jupyter_port, servers_only=False))
    return killed
    

def kill_all_vivarium_processes(grpc_port=50051, interface_port=5006, include_clients=True):
    """Forcefully kill all Vivarium-related processes.

    This function aggressively cleans up any remaining processes by:
    1. Killing server and interface processes by name
    2. Killing any process listening on the gRPC port (default 50051)
    3. Killing any process listening on the interface port (default 5006)

    Use this when normal cleanup fails or to ensure a clean state.

    Args:
        grpc_port: Port used by the gRPC server (default 50051)
        interface_port: Port used by the Panel interface (default 5006)
        include_clients: If True (default), also kill client connections to the ports.
                        Set to False when calling from tests to avoid killing the test process.

    Returns:
        dict with 'by_name' and 'by_port' keys listing killed PIDs
    """
    killed = {'by_name': [], 'by_port': []}

    # Kill by process name
    interface_pids, server_pids = get_server_interface_pids()
    if interface_pids:
        terminate_process(interface_pids)
        killed['by_name'].extend(interface_pids)
    if server_pids:
        terminate_process(server_pids)
        killed['by_name'].extend(server_pids)

    # Give processes time to terminate
    time.sleep(0.5)

    # Kill processes on the ports (servers only if include_clients=False)
    killed['by_port'].extend(kill_port_processes(grpc_port, servers_only=not include_clients))
    killed['by_port'].extend(kill_port_processes(interface_port, servers_only=not include_clients))

    total = len(killed['by_name']) + len(killed['by_port'])
    if total > 0:
        lg.info(f"Killed {total} Vivarium process(es)")
    else:
        lg.info("No Vivarium processes found")

    return killed


def start_process_and_parse_url(process_command, show_output=True, timeout=10):
    """Start a process and parse URL from its output

    :param process_command: command to start the process
    :param show_output: whether to echo subprocess stdout/stderr
    :param timeout: seconds to wait for URL to appear in output
    :return: tuple of (Popen process object, URL string or None)
    """
    import threading

    process = subprocess.Popen(
        process_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url_found = [None]  # Use list to allow modification in thread

    def read_output():
        url_pattern = re.compile(r'(http://[^\s]+)')
        try:
            for line in process.stdout:
                if show_output:
                    print(line, end='')
                if url_found[0] is None:
                    match = url_pattern.search(line)
                    if match:
                        url_found[0] = match.group(1)
        except:
            pass

    # Start thread to read output
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

    # Wait for URL with timeout
    start_time = time.time()
    while url_found[0] is None and time.time() - start_time < timeout:
        if process.poll() is not None:  # Process terminated
            break
        time.sleep(0.1)

    return process, url_found[0]


def wait_for_grpc_server(host="localhost", port=50051, timeout=30.0, poll_interval=0.5):
    """
    Wait for the gRPC server to be ready using the standard health checking protocol.

    Args:
        host: Server hostname
        port: Server port
        timeout: Maximum seconds to wait
        poll_interval: Seconds between connection attempts

    Returns:
        True if server is ready, False if timeout reached
    """
    channel = grpc.insecure_channel(f"{host}:{port}")
    health_stub = health_pb2_grpc.HealthStub(channel)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            request = health_pb2.HealthCheckRequest(service="")
            response = health_stub.Check(request, timeout=1.0)
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                channel.close()
                return True
        except grpc.RpcError:
            pass
        time.sleep(poll_interval)

    channel.close()
    return False


def check_server_running(host="localhost", port=50051):
    """Check if the gRPC server is currently running.

    Args:
        host: Server hostname
        port: Server port

    Returns:
        True if server is running and responding to health checks, False otherwise
    """
    return wait_for_grpc_server(host=host, port=port, timeout=1.0, poll_interval=0.2)


def wait_for_http(url, retries=1, delay=5.0, timeout=10.0):
    """Wait for an HTTP endpoint to respond with status 200.

    Useful for checking if a web server (e.g., Panel interface) is ready.
    Can be used programmatically or from command line for CI health checks.

    Args:
        url: The URL to check (e.g., 'http://localhost:5006/run_interface')
        retries: Maximum number of attempts (default: 1, no retries)
        delay: Seconds to wait between retries (default: 5.0)
        timeout: Timeout in seconds for each HTTP request (default: 10.0)

    Returns:
        True if URL responded with 200, False otherwise

    Example:
        # Programmatic use
        if wait_for_http('http://localhost:5006/run_interface', retries=12, delay=5):
            print("Panel is ready!")

        # CI use (will exit with code 1 on failure)
        python -c "from vivarium.utils.handle_server_interface import wait_for_http; \\
                   assert wait_for_http('http://localhost:5006/run_interface', retries=12)"
    """
    import urllib.request
    import urllib.error

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    lg.info(f"HTTP check OK: {url} (attempt {attempt}/{retries})")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            lg.debug(f"HTTP check attempt {attempt}/{retries} for {url}: {e}")

        if attempt < retries:
            time.sleep(delay)

    lg.warning(f"HTTP check failed after {retries} attempts: {url}")
    return False


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

    cmd_args = [f"scene={scene_name}"] if scene_name else []
    server_command = get_server_command(cmd_args)

    lg.info(f"Starting Vivarium server{f' with scene {scene_name!r}' if scene_name else ''}...")

    server_process = subprocess.Popen(
        server_command,
        stdout=None if show_output else subprocess.DEVNULL,
        stderr=None if show_output else subprocess.DEVNULL
    )

    # Wait for gRPC server to be ready
    if not wait_for_grpc_server(timeout=timeout):
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

    try:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            lg.info("Server terminated gracefully")
        except subprocess.TimeoutExpired:
            lg.warning("Server did not terminate gracefully, forcing kill...")
            server_process.kill()
            server_process.wait(timeout=3)
            lg.info("Server killed")
    except Exception as e:
        lg.warning(f"Error stopping server: {e}")


def start_panel_interface(allow_external_origins=False, show_output=True, timeout=10):
    """Start only the Panel web interface (assumes server is running).

    Args:
        allow_external_origins: Whether to allow websocket connections from
                               external origins (e.g., ngrok).
        show_output: Whether to show interface output in console.
        timeout: Seconds to wait for URL to appear in output.

    Returns:
        tuple: (Popen process, interface_url or None)
    """
    interface_command = get_interface_command(allow_external_origins)
    lg.info("Starting web interface...")

    interface_process, interface_url = start_process_and_parse_url(
        interface_command,
        show_output=show_output,
        timeout=timeout
    )

    if interface_url:
        lg.info(f"Interface available at: {interface_url}")
    else:
        # If we can't parse the URL from output, construct it
        interface_url = "http://localhost:5006/run_interface"
        lg.info(f"Interface should be available at: {interface_url}")

    return interface_process, interface_url


def stop_panel_interface(interface_process):
    """Stop a Panel interface process.

    Args:
        interface_process: Popen process object to terminate
    """
    if interface_process is None:
        return

    lg.info(f"Stopping Panel interface (PID: {interface_process.pid})...")

    try:
        interface_process.terminate()
        try:
            interface_process.wait(timeout=5)
            lg.info("Interface terminated gracefully")
        except subprocess.TimeoutExpired:
            lg.warning("Interface did not terminate gracefully, forcing kill...")
            interface_process.kill()
            interface_process.wait(timeout=3)
            lg.info("Interface killed")
    except Exception as e:
        lg.warning(f"Error stopping interface: {e}")


def get_ngrok_token():
    """
    Get ngrok token from environment variable or Colab secrets.
    
    Returns:
        str: The ngrok auth token
        
    Raises:
        RuntimeError: If no token is found
    """
    import sys
    
    # First try environment variable
    token = os.environ.get('NGROK_TOKEN')
    if token:
        return token
    
    # Then try Colab secrets
    if 'google.colab' in sys.modules:
        try:
            from google.colab import userdata
            token = userdata.get('NGROK_TOKEN')
            if token:
                return token
        except Exception:
            pass
    
    # No token found, raise helpful error
    raise RuntimeError(
        "NGROK_TOKEN not found. Set it via:\n"
        "  - Environment variable: export NGROK_TOKEN=your_token\n"
        "  - Colab secrets: Add 'NGROK_TOKEN' in the key icon (🔑) sidebar\n\n"
        "Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken"
    )


def create_ngrok_tunnel(port=5006, token=None):
    """
    Create an ngrok tunnel to expose a local port publicly.
    
    Args:
        port: Local port to tunnel (default: 5006 for Panel)
        token: ngrok auth token. If None, reads from NGROK_TOKEN env var or Colab secrets.
    
    Returns:
        str: The public ngrok URL
        
    Raises:
        RuntimeError: If pyngrok is not installed or token is missing
    """
    try:
        from pyngrok import ngrok
    except ImportError:
        raise RuntimeError(
            "pyngrok is not installed. Install it with: pip install pyngrok"
        )
    
    # Get token
    if token is None:
        token = get_ngrok_token()
    
    ngrok.set_auth_token(token)
    
    print(f"🔗 Creating ngrok tunnel for port {port}...")
    public_url = ngrok.connect(port, bind_tls=True)
    ngrok_url = public_url.public_url
    print(f"✓ Public URL: {ngrok_url}")
    
    return ngrok_url


def close_ngrok_tunnel():
    """
    Close all ngrok tunnels.
    
    Safe to call even if no tunnel is active.
    """
    try:
        from pyngrok import ngrok
        ngrok.disconnect_all()
        ngrok.kill()
        print("✓ Ngrok tunnel closed")
    except ImportError:
        pass  # pyngrok not installed, nothing to close
    except Exception:
        pass  # Tunnel may not have been active


def check_colab_environment():
    """
    Check if running in Google Colab and validate ngrok token is configured.
    
    Call this before start_session(ngrok=True) in Colab to get helpful setup instructions
    if the token is missing.
    
    Raises:
        RuntimeError: If not in Colab or NGROK_TOKEN is not configured
    """
    import sys
    
    if 'google.colab' not in sys.modules:
        raise RuntimeError("This function is only for Google Colab environment")
    
    try:
        get_ngrok_token()
        print("✓ NGROK_TOKEN found in Colab Secrets")
    except RuntimeError:
        print("\nIn order to run Vivarium in Colab, you need to use ngrok to enable access to the web interface.")
        print("Here are the steps to set up your ngrok token:")
        print("1. Create an account on ngrok: https://dashboard.ngrok.com/signup")
        print("2. Once you are logged in, go to the 'Auth' section: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("3. Copy your authtoken")
        print("4. In this Colab notebook, click the key icon (🔑) in the left sidebar")
        print("5. Click 'Add a new secret'")
        print("6. Set Name: NGROK_TOKEN")
        print("7. Paste your authtoken as the Value")
        print("8. Toggle on 'Notebook access' for this notebook")
        print("9. Re-run this cell\n")
        raise RuntimeError("NGROK_TOKEN secret not configured")


if __name__ == "__main__":
    platform = os.name
    print(f"Platform: {platform}")
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=False)
    start_simulation_server("session_1")
    start_panel_interface()
