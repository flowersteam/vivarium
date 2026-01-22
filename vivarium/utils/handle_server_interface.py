import os
import time
import psutil
import multiprocessing
import subprocess
import signal
import logging
import re
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


lg = logging.getLogger(__name__)

SERVER_PROCESS_NAME = "scripts/run_server.py"
INTERFACE_PROCESS_NAME = "scripts/run_interface.py"
SERVER_PROCESS_NAME_WIN = "scripts\\run_server.py"
INTERFACE_PROCESS_NAME_WIN = "scripts\\run_interface.py"


def start_jupyter_server(port=8889, notebook_dir=None, show_output=True, return_process_object=False):
    """Start a Jupyter notebook server with iframe-friendly configuration

    :param port: Port to run Jupyter on, defaults to 8889
    :param notebook_dir: Directory to start Jupyter in, defaults to project root
    :param show_output: Whether to show Jupyter server output
    :param return_process_object: If True, return Popen object instead of multiprocessing.Process
    :return: Process object (Popen or multiprocessing.Process)
    :raises RuntimeError: If the requested port is already in use
    """
    # Check if the requested port is already in use
    if check_jupyter_running(port):
        raise RuntimeError(
            f"Port {port} is already in use. Please stop the existing Jupyter server or choose a different port."
        )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    if notebook_dir is None:
        notebook_dir = project_root

    config_path = os.path.join(project_root, "vivarium/interface/jupyter_config_iframe.py")

    jupyter_command = [
        "jupyter",
        "notebook",
        f"--config={config_path}",
        f"--port={port}",
        f"--notebook-dir={notebook_dir}",
        "--no-browser",
    ]

    lg.info(f"Starting Jupyter notebook server on port {port}...")
    lg.info(f"Notebook directory: {notebook_dir}")

    if return_process_object:
        # Return a Popen object for direct process management
        jupyter_process = subprocess.Popen(
            jupyter_command,
            stdout=None if show_output else subprocess.DEVNULL,
            stderr=None if show_output else subprocess.DEVNULL
        )
        lg.info(f"Jupyter server started (PID: {jupyter_process.pid})")
    else:
        # Return a multiprocessing.Process for background execution
        jupyter_process = multiprocessing.Process(
            target=subprocess.run,
            args=(jupyter_command,),
            kwargs={"stdout": None if show_output else subprocess.DEVNULL,
                    "stderr": None if show_output else subprocess.DEVNULL}
        )
        jupyter_process.start()
        lg.info(f"Jupyter server started (PID: {jupyter_process.pid})")

    lg.info(f"Access it at: http://localhost:{port}")

    return jupyter_process


def check_jupyter_running(port=8889):
    """Check if a Jupyter server is running on the specified port

    :param port: Port to check
    :return: True if Jupyter is running, False otherwise
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result == 0
    except Exception:
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
                # For multiprocessing.Process
                jupyter_process.terminate()
                jupyter_process.join(timeout=3)
                if jupyter_process.is_alive():
                    jupyter_process.kill()
                killed = True
                lg.info("Jupyter server terminated via multiprocessing.Process")
        except Exception as e:
            lg.warning(f"Failed to kill Jupyter via process object: {e}")

    # Always check port and kill any remaining process (Jupyter can fork)
    lg.info(f"Checking if Jupyter is still running on port {port}...")
    if check_jupyter_running(port):
        lg.warning(f"Jupyter still detected on port {port}, force killing...")
        try:
            # Find PID listening on the port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split('\n')
            lg.info(f"Found PIDs on port {port}: {pids}")
            for pid in pids:
                if pid and pid.strip():
                    try:
                        # Use SIGKILL (9) to force kill Jupyter
                        os.kill(int(pid), signal.SIGKILL)
                        lg.info(f"Force killed Jupyter process with PID: {pid}")
                        killed = True
                    except Exception as e:
                        lg.warning(f"Failed to kill PID {pid}: {e}")
        except Exception as e:
            lg.warning(f"Failed to find/kill Jupyter by port: {e}")
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


def start_process(process_command, url_queue=None, show_output=True):
    """Start a process with the given command

    :param process_command: command to start the process
    :param url_queue: optional Queue to send the URL back to parent process
    :param show_output: whether to echo subprocess stdout/stderr
    """
    if url_queue is None:
        subprocess.run(process_command, stdout=None if show_output else subprocess.DEVNULL, stderr=None if show_output else subprocess.DEVNULL)
    else:
        # Capture output to extract URL
        process = subprocess.Popen(
            process_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Parse output for URL
        url_pattern = re.compile(r'(http://[^\s]+)')
        for line in process.stdout:
            if show_output:
                print(line, end='')  # Still print to console
            match = url_pattern.search(line)
            if match:
                url = match.group(1)
                url_queue.put(url)
        
        process.wait()


# Define parameters of the simulator
def start_server_and_interface(
    cmd_args, start_interface: bool = True, server_timeout: float = 30.0, safe_mode=True, show_output=True, allow_external_origins=False
):
    """Start the server and interface for the given scene

    :param cmd_args: command line arguments to pass to the server script
    :param start_interface: whether to start the interface, defaults to True
    :param server_timeout: maximum seconds to wait for gRPC server to be ready
    :param safe_mode: whether to prompt before stopping existing processes
    :param allow_external_origins: whether to allow websocket connections from external origins (e.g., ngrok)
    :return: URL of the interface if started, None otherwise
    :raises RuntimeError: if gRPC server doesn't start within timeout
    """
    if os.name == "nt":
        lg.warning(
            "The 'start_server_and_interface' function is not supported on Windows OS"
        )
        lg.warning(
            "Instead, start the server and interface by running the following command from the root directory in a Windows Powershell (make sure to activate the virtual environment before). Then click on the link to open the web interface:"
        )
        lg.warning(f"\nstart_all.bat {cmd_args[0] if cmd_args else ''}")
        return None

    # first ensure no interface or server is running
    processes_running = stop_server_and_interface(safe_mode=safe_mode)

    if processes_running:
        lg.warning(
            "\nServer and Interface processes are still running, please stop them before starting new ones"
        )
        lg.warning("ERROR: New processes will not be started")
        return None

    # find the path to the server and interface scripts
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    server_script = os.path.join(project_root, SERVER_PROCESS_NAME)
    interface_script = os.path.join(project_root, INTERFACE_PROCESS_NAME)

    server_command = ["python3", server_script, *cmd_args]

    print("\n🚀 Starting Vivarium server...")
    server_process = multiprocessing.Process(
        target=start_process, args=(server_command,), kwargs={"show_output": show_output}
    )
    server_process.start()
    
    # Wait for gRPC server to be ready
    if not wait_for_grpc_server(timeout=server_timeout):
        raise RuntimeError(f"gRPC server did not start within {server_timeout} seconds")
    
    interface_url = None
    if start_interface:

        interface_command = [
            "panel",
            "serve",
            interface_script,
        ]

        # Allow external origins (e.g., ngrok) if requested
        if allow_external_origins:
            interface_command.append("--allow-websocket-origin=*")

        interface_command.append("--args")

        # Create a queue to receive the URL from the subprocess
        url_queue = multiprocessing.Queue()

        # start the interface
        print("\n🌐 Starting web interface...")
        interface_process = multiprocessing.Process(
            target=start_process, args=(interface_command, url_queue), kwargs={"show_output": show_output}
        )
        interface_process.start()
        
        # Wait for URL with timeout
        try:
            interface_url = url_queue.get(timeout=10)
            print(f"\n✓ Interface available at: {interface_url}")
        except:
            # If we can't get the URL from the queue, construct it
            interface_url = "http://localhost:5006/run_interface"
            print(f"\n✓ Interface should be available at: {interface_url}")
    
    return interface_url


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
    start_server_and_interface("session_1", notebook_mode=True, wait_time=6)
