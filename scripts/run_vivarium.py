import logging
import argparse
import webbrowser
import time

from vivarium.utils.handle_server_interface import (
    get_server_interface_pids,
    start_server_and_interface,
    stop_server_and_interface,
)


lg = logging.getLogger(__name__)




def main(cmd_args):
    """Main function to start Vivarium

    :param cmd_args: Command line arguments to pass to the server
    """
    interface_pids, server_pids = get_server_interface_pids()
    print(f"Interface PIDs: {interface_pids}")
    print(f"Server PIDs: {server_pids}")
    stop_server_and_interface(safe_mode=True)

    start_server_and_interface(cmd_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Vivarium simulator with web interface"
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Scene configuration to load (e.g., session_1, prey_predator)"
    )

    args = parser.parse_args()

    # Build cmd_args for the server with proper Hydra syntax
    cmd_args = [f"scene={args.scene}"] if args.scene else []

    main(cmd_args)

    webbrowser.open("http://localhost:5006/run_interface")

    try:
        # Keep the script running to allow the servers to run
        print("\nVivarium is running. Press Ctrl+C to stop.\n")
        print("💡 Tip: Start Jupyter from the Notebook tab in the web interface\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping Vivarium...")

        # Stop server and interface
        stop_server_and_interface(safe_mode=False)
        print("\n✓ Vivarium stopped. Goodbye!\n")
        print("Note: Jupyter servers (if running) are not stopped automatically.")
