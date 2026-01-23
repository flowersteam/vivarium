import argparse

import panel as pn
from vivarium.interface.panel_app import WindowManager
from vivarium.utils.runtime import is_frozen


parser = argparse.ArgumentParser(description="Run the Vivarium interface.")
parser.add_argument(
    "--notebook_mode", type=str, default="False", help="Run in notebook mode."
)
args = parser.parse_args()

if args.notebook_mode == "True":
    notebook_mode = True
elif args.notebook_mode == "False":
    notebook_mode = False
else:
    raise ValueError(
        f"Invalid value for notebook_mode: {args.notebook_mode}. Use either 'True' or 'False'."
    )

# If running as PyInstaller bundle, start the server programmatically
# Otherwise, mark as servable for `panel serve` to handle
if is_frozen():
    # Frozen mode - start Panel server programmatically
    # Pass a function that creates the app so it's created after event loop starts
    def create_app():
        wm = WindowManager(notebook_mode=notebook_mode)
        return wm.app

    pn.serve(
        {'/run_interface': create_app},
        port=5006,
        title="Vivarium",
        show=False,  # Don't auto-open browser
        threaded=False,  # Block until server stops
    )
else:
    # Development mode - create app and mark as servable for `panel serve` command
    wm = WindowManager(notebook_mode=notebook_mode)
    wm.app.servable(title="Vivarium")
