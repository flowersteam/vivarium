import os
import time
import psutil
import multiprocessing
import subprocess
import signal
import logging

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


def start_process(process_command):
    """Start a process with the given command

    :param process_command: command to start the process
    """
    subprocess.run(process_command)


# Define parameters of the simulator
def start_server_and_interface(
    scene_name: str, notebook_mode: bool = True, wait_time: int = 7, safe_mode=True
):
    """Start the server and interface for the given scene

    :param scene_name: scene name
    :param notebook_mode: notebook_mode to adapt the interface, defaults to True
    """
    if os.name == "nt":
        lg.warning(
            "The 'start_server_and_interface' function is not supported on Windows OS"
        )
        lg.warning(
            "Instead, start the server and interface by running the following command from the root directory in a Windows Powershell (make sure to activate the virtual environment before). Then click on the link to open the web interface:"
        )
        lg.warning(f"\n.\start_all.bat {scene_name}")
        return

    # first ensure no interface or server is running
    processes_running = stop_server_and_interface(safe_mode=safe_mode)

    if processes_running:
        lg.warning(
            "\nServer and Interface processes are still running, please stop them before starting new ones"
        )
        lg.warning("ERROR: New processes will not be started")
        return

    # find the path to the server and interface scripts
    print(f"{os.path.dirname(__file__)}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    server_script = os.path.join(project_root, SERVER_PROCESS_NAME)
    interface_script = os.path.join(project_root, INTERFACE_PROCESS_NAME)

    server_command = ["python3", server_script, f"scene={scene_name}"]

    print("\nSTARTING SERVER")
    server_process = multiprocessing.Process(
        target=start_process, args=(server_command,)
    )
    server_process.start()
    time.sleep(wait_time)

    interface_command = [
        "panel",
        "serve",
        interface_script,
        "--args",
        f"--notebook_mode={str(notebook_mode)}",
    ]

    # start the interface
    print("\nSTARTING INTERFACE")
    interface_process = multiprocessing.Process(
        target=start_process, args=(interface_command,)
    )
    interface_process.start()


if __name__ == "__main__":
    platform = os.name
    print(f"Platform: {platform}")
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=False)
    start_server_and_interface("session_1", notebook_mode=True, wait_time=6)
