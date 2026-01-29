import argparse
import signal
import sys
import atexit
import logging

import panel as pn
from vivarium.interface.panel_app import WindowManager
from vivarium.utils.handle_server_interface import kill_vivarium_processes

lg = logging.getLogger(__name__)

# Track if cleanup has already run (avoid double cleanup)
_cleanup_done = False


def cleanup():
    """Clean up all Vivarium processes when the interface exits."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    lg.info("Cleaning up Vivarium processes...")
    try:
        # Kill the gRPC server, its clients, and Jupyter if running
        # Don't kill interface (that's us, already exiting)
        killed = kill_vivarium_processes(server=True, clients=True, jupyter=True, interface=False)
        if killed:
            lg.info(f"Stopped {len(killed)} process(es)")
    except Exception as e:
        lg.warning(f"Cleanup error: {e}")


def signal_handler(signum, _frame):
    """Handle termination signals gracefully."""
    sig_name = signal.Signals(signum).name
    lg.info(f"Received {sig_name}, shutting down...")
    cleanup()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Also register atexit handler as fallback
atexit.register(cleanup)


parser = argparse.ArgumentParser(description="Run the Vivarium interface.")
parser.add_argument(
    "--dont-open-browser", action="store_true", help="Don't open the interface in a browser window."
)
parser.add_argument(
    "--allow-external-origins", action="store_true", help="Allow websocket connections from external origins (e.g., ngrok)."
)
parser.add_argument(
    "--no-cleanup", action="store_true", help="Don't kill server/jupyter when the interface exits (useful for development)."
)
args = parser.parse_args()

# Disable cleanup if requested
if args.no_cleanup:
    _cleanup_done = True  # Prevents cleanup from running


def create_app():
    wm = WindowManager()
    return wm.app


serve_kwargs = {
    'port': 5006,
    'title': "Vivarium",
    'show': not args.dont_open_browser,
    'threaded': False,  # Block until server stops
}

if args.allow_external_origins:
    serve_kwargs['websocket_origin'] = '*'

pn.serve({'/run_interface': create_app}, **serve_kwargs)
