import argparse

from vivarium.interface.panel_app import WindowManager

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

# Serve the app to launch the interface
wm = WindowManager(notebook_mode=notebook_mode)
wm.app.servable(title="Vivarium")
