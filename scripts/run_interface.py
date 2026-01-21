import argparse

from vivarium.interface.panel_app import WindowManager

# import sys

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description="Run the Vivarium interface.")
parser.add_argument(
    "--notebook_mode", type=str, default="False", help="Run in notebook mode."
)
parser.add_argument(
    "--jupyter_enabled", type=str, default="False", help="Whether Jupyter server is running."
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

if args.jupyter_enabled == "True":
    jupyter_enabled = True
elif args.jupyter_enabled == "False":
    jupyter_enabled = False
else:
    raise ValueError(
        f"Invalid value for jupyter_enabled: {args.jupyter_enabled}. Use either 'True' or 'False'."
    )

# Serve the app to launch the interface
wm = WindowManager(notebook_mode=notebook_mode, jupyter_enabled=jupyter_enabled)
wm.app.servable(title="Vivarium")
