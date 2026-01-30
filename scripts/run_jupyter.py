"""Jupyter server launcher for PyInstaller builds.

This script starts a Jupyter notebook server programmatically,
allowing it to be bundled as a standalone executable.
"""
import argparse
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from notebook.notebookapp import NotebookApp


def main():
    parser = argparse.ArgumentParser(description='Start Jupyter notebook server')
    parser.add_argument('--port', type=int, default=8889, help='Port to run Jupyter on')
    parser.add_argument('--notebook-dir', type=str, default=None, help='Directory to start Jupyter in')
    parser.add_argument('--config', type=str, default=None, help='Path to Jupyter config file')
    args = parser.parse_args()

    app = NotebookApp.instance()
    app.port = args.port
    if args.notebook_dir:
        app.notebook_dir = args.notebook_dir
    app.open_browser = False

    # iframe-friendly settings (same as jupyter_config_iframe.py)
    app.allow_origin = '*'
    app.disable_check_xsrf = True
    app.tornado_settings = {
        'headers': {
            'Content-Security-Policy': "frame-ancestors 'self' http://localhost:* http://127.0.0.1:*"
        }
    }

    # Set default kernel name
    # Note: In PyInstaller builds, the bundled Python acts as the kernel
    app.kernel_manager_class = 'notebook.services.kernels.kernelmanager.MappingKernelManager'

    if args.config:
        app.load_config_file(args.config)

    app.initialize()
    app.start()


if __name__ == '__main__':
    main()
