import os
import time
import psutil
import multiprocessing
import subprocess
import signal
import logging
import re


lg = logging.getLogger(__name__)

SERVER_PROCESS_NAME = "scripts/run_server.py"
INTERFACE_PROCESS_NAME = "scripts/run_interface.py"
SERVER_PROCESS_NAME_WIN = "scripts\\run_server.py"
INTERFACE_PROCESS_NAME_WIN = "scripts\\run_interface.py"


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
        print("\nStopping server and interface processes\n")
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
            lg.warning("\nServer and Interface processes have been stopped\n")

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
    cmd_args, start_interface: bool = True, wait_time: int = 7, safe_mode=True, show_output=True
):
    """Start the server and interface for the given scene

    :param cmd_args: command line arguments to pass to the server script
    :param start_interface: whether to start the interface, defaults to True
    :param wait_time: time to wait for server startup before starting interface
    :param safe_mode: whether to prompt before stopping existing processes
    :return: URL of the interface if started, None otherwise
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
    
    interface_url = None
    if start_interface:
        time.sleep(wait_time)

        interface_command = [
            "panel",
            "serve",
            interface_script,
            "--args",
        ]

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


def setup_colab_environment(scene,
                            branch="main", 
                            port=5006,
                            startup_delay=10):
    """
    Set up Vivarium in Google Colab with ngrok tunnel.
    
    Args:
        branch: Git branch to clone
        scene: Scene name for the simulator
        port: Port for Panel server
        startup_delay: Seconds to wait for server startup
    
    Returns:
        tuple: (controller, WindowManager, ngrok_url)
    """
    import sys
    import subprocess
    
    if 'google.colab' not in sys.modules:
        raise RuntimeError("This function is only for Google Colab environment")
    
    from google.colab import userdata
    
    try:
        ngrok_token = userdata.get('NGROK_TOKEN')
        print("✓ Using ngrok token from Colab Secrets")
    except Exception:
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
    
    # Start server in background
    print("🚀 Starting Vivarium server...")
    server_cmd = f"python /content/vivarium/scripts/run_server.py scene={scene}"
    subprocess.Popen(
        server_cmd.split(),
        stdout=open('/content/vivarium_server.log', 'w'),
        stderr=subprocess.STDOUT
    )    

    subprocess.run(["pip", "install", "pyngrok", "jupyter_bokeh", "-q"], check=True)
    import panel as pn
    from pyngrok import ngrok
    import nest_asyncio    
    
    # Configure environment
    nest_asyncio.apply()
    pn.extension(inline=True)
    ngrok.set_auth_token(ngrok_token)
    
    from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
    from vivarium.controllers import VivariumController
    from vivarium.interface.panel_app import WindowManager

    # Wait for server and initialize
    print(f"⏳ Waiting {startup_delay}s for server startup...")
    
    time.sleep(startup_delay)    
    print("🔧 Initializing controller...")
    client = SimulatorGRPCClient()
    controller = VivariumController.start_session(scene_name=client.scene_name, client=client)
    wm = WindowManager()
    
    # Start Panel server with ngrok
    print("🌐 Starting Panel server and ngrok tunnel...")
    server = pn.serve(
        wm.app,
        port=port,
        threaded=True,
        show=False,
        websocket_origin="*"
    )
    
    public_url = ngrok.connect(port, bind_tls=True)
    ngrok_url = public_url.public_url
    
    print(f"\n✅ Setup complete!")
    
    return controller, wm, ngrok_url


if __name__ == "__main__":
    platform = os.name
    print(f"Platform: {platform}")
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=False)
    start_server_and_interface("session_1", notebook_mode=True, wait_time=6)
