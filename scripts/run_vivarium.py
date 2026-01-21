import logging
import argparse
import webbrowser
import os
import multiprocessing
import subprocess

from vivarium.utils.handle_server_interface import (
    get_server_interface_pids,
    start_server_and_interface,
    stop_server_and_interface,
)


lg = logging.getLogger(__name__)


def start_jupyter_server(port=8889, notebook_dir=None, show_output=True):
    """Start a Jupyter notebook server with iframe-friendly configuration

    :param port: Port to run Jupyter on, defaults to 8889
    :param notebook_dir: Directory to start Jupyter in, defaults to project root
    :param show_output: Whether to show Jupyter server output
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

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

    print(f"\n📓 Starting Jupyter notebook server on port {port}...")
    print(f"   Notebook directory: {notebook_dir}")

    jupyter_process = multiprocessing.Process(
        target=subprocess.run,
        args=(jupyter_command,),
        kwargs={"stdout": None if show_output else subprocess.DEVNULL,
                "stderr": None if show_output else subprocess.DEVNULL}
    )
    jupyter_process.start()

    print(f"✓ Jupyter server started (PID: {jupyter_process.pid})")
    print(f"  Access it at: http://localhost:{port}")

    return jupyter_process


def main(cmd_args, start_jupyter=False, jupyter_port=8889, jupyter_notebook_dir=None):
    """Main function to start Vivarium and optionally Jupyter

    :param cmd_args: Command line arguments to pass to the server
    :param start_jupyter: Whether to start Jupyter server
    :param jupyter_port: Port for Jupyter server
    :param jupyter_notebook_dir: Directory for Jupyter notebooks
    """
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=True)

    # Start Jupyter if requested
    jupyter_process = None
    if start_jupyter:
        jupyter_process = start_jupyter_server(
            port=jupyter_port,
            notebook_dir=jupyter_notebook_dir
        )

    start_server_and_interface(cmd_args, jupyter_enabled=start_jupyter)

    return jupyter_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Vivarium simulator with web interface and optional Jupyter notebook server"
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Scene configuration to load (e.g., session_1, prey_predator)"
    )
    parser.add_argument(
        "--jupyter",
        action="store_true",
        help="Start a Jupyter notebook server alongside Vivarium"
    )
    parser.add_argument(
        "--jupyter-port",
        type=int,
        default=8889,
        help="Port for Jupyter notebook server (default: 8889)"
    )
    parser.add_argument(
        "--jupyter-dir",
        type=str,
        default=None,
        help="Directory for Jupyter notebooks (default: project root)"
    )

    args = parser.parse_args()

    # Build cmd_args for the server with proper Hydra syntax
    cmd_args = [f"scene={args.scene}"] if args.scene else []

    jupyter_process = main(
        cmd_args,
        start_jupyter=args.jupyter,
        jupyter_port=args.jupyter_port,
        jupyter_notebook_dir=args.jupyter_dir
    )

    webbrowser.open("http://localhost:5006/run_interface")
    # try:
    #     # Keep the script running to allow the server to run
    #     while True:
    #         pass
    # except KeyboardInterrupt:
    #     lg.info("Script stopped by user. Cleaning up...")
    #     stop_server_and_interface(safe_mode=True)
