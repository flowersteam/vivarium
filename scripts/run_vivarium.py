import logging
import argparse
import webbrowser
import time
import sys
import os

from vivarium.utils.handle_server_interface import (
    get_server_interface_pids,
    stop_server_and_interface,
    start_simulation_server,
    start_panel_interface,
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

    # Extract scene name from cmd_args (format: "scene=name")
    scene_name = None
    for arg in cmd_args:
        if arg.startswith("scene="):
            scene_name = arg.replace("scene=", "")
            break
    print("\n🚀 Starting Vivarium server...")
    start_simulation_server(scene_name)
    print("\n🌐 Starting web interface...")
    start_panel_interface()


if __name__ == "__main__":
    # CRITICAL: PyInstaller safety guards to prevent recursive process spawning

    # Guard 1: Detect if running as a frozen PyInstaller executable
    if getattr(sys, 'frozen', False):
        # We're running as a PyInstaller bundle

        # Guard 2: Check if this is already a child process
        if os.environ.get('VIVARIUM_CHILD_PROCESS') == '1':
            # This is a child process that was spawned by a frozen parent
            # Don't execute main logic to prevent infinite recursion
            print("WARNING: Child process detected, exiting to prevent recursive spawning.")
            sys.exit(0)

        # Guard 3: Mark that we're in a frozen environment
        # Child processes spawned from here will inherit this flag
        os.environ['VIVARIUM_FROZEN'] = '1'

    # Guard 4: Spawn limit counter (emergency brake)
    # This will catch any edge cases where the above guards fail
    _spawn_count_file = os.path.join(os.path.expanduser('~'), '.vivarium_spawn_count')
    MAX_SPAWN_COUNT = 5

    try:
        if os.path.exists(_spawn_count_file):
            with open(_spawn_count_file, 'r') as f:
                spawn_count = int(f.read().strip())
            spawn_count += 1
        else:
            spawn_count = 1

        if spawn_count > MAX_SPAWN_COUNT:
            print(f"ERROR: Spawn limit exceeded ({spawn_count}). Preventing potential fork bomb.")
            print("If you see this error repeatedly, please report it as a bug.")
            # Clean up the counter file
            if os.path.exists(_spawn_count_file):
                os.remove(_spawn_count_file)
            sys.exit(1)

        # Write updated count
        with open(_spawn_count_file, 'w') as f:
            f.write(str(spawn_count))
    except Exception:
        # If we can't read/write the counter, continue anyway
        pass

    parser = argparse.ArgumentParser(
        description="Run Vivarium simulator with web interface"
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Scene configuration to load (e.g., session_1, prey_predator)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically"
    )

    args = parser.parse_args()

    # Build cmd_args for the server with proper Hydra syntax
    cmd_args = [f"scene={args.scene}"] if args.scene else []

    main(cmd_args)

    if not args.no_browser:
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
    finally:
        # Clean up spawn counter when exiting normally
        try:
            if os.path.exists(_spawn_count_file):
                os.remove(_spawn_count_file)
        except Exception:
            pass
