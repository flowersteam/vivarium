"""
Process management for Vivarium server, interface, and Jupyter processes.

Provides PID lookup, kill, and terminate functions for all Vivarium
subprocess types. Cross-platform (Unix/Windows).
"""

import os
import signal
import subprocess
import time
import logging

import psutil

from vivarium.runtime.paths import DEFAULT_JUPYTER_PORT


lg = logging.getLogger(__name__)

SERVER_PROCESS_NAME = "scripts/run_server.py"
INTERFACE_PROCESS_NAME = "scripts/run_interface.py"
SERVER_PROCESS_NAME_WIN = "run_server.py"
INTERFACE_PROCESS_NAME_WIN = "run_interface.py"


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
                            grpc_port=50051, interface_port=5006, jupyter_port=DEFAULT_JUPYTER_PORT,
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
    from vivarium.runtime._jupyter import get_started_jupyter_ports

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
