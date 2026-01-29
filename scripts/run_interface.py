import argparse

import panel as pn
from vivarium.interface.panel_app import WindowManager

parser = argparse.ArgumentParser(description="Run the Vivarium interface.")
parser.add_argument(
    "--dont-open-browser", action="store_true", help="Don't open the interface in a browser window."
)
parser.add_argument(
    "--allow-external-origins", action="store_true", help="Allow websocket connections from external origins (e.g., ngrok)."
)
args = parser.parse_args()    
    
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
