"""Jupyter server launcher for PyInstaller builds.

This script starts a Jupyter notebook server programmatically,
allowing it to be bundled as a standalone executable.

Uses notebook 7.x API (JupyterNotebookApp) which is built on jupyter_server.
"""
import argparse
import sys

from notebook.app import JupyterNotebookApp

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser(description='Start Jupyter notebook server')
    parser.add_argument('--port', type=int, default=8889, help='Port to run Jupyter on')
    parser.add_argument('--notebook-dir', type=str, default=None, help='Directory to start Jupyter in')
    parser.add_argument('--config', type=str, default=None, help='Path to Jupyter config file')
    args = parser.parse_args()

    # Build argv for JupyterNotebookApp
    argv = [
        f'--port={args.port}',
        '--no-browser',
        # iframe-friendly settings
        '--ServerApp.allow_origin=*',
        '--ServerApp.disable_check_xsrf=True',
        '--ServerApp.tornado_settings={"headers": {"Content-Security-Policy": "frame-ancestors \'self\' http://localhost:* http://127.0.0.1:*"}}',
    ]

    if args.notebook_dir:
        argv.append(f'--notebook-dir={args.notebook_dir}')

    if args.config:
        argv.append(f'--config={args.config}')

    # Use notebook 7.x API
    JupyterNotebookApp.launch_instance(argv=argv)


if __name__ == '__main__':
    main()
