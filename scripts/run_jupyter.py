"""Jupyter server and kernel launcher for PyInstaller builds.

This script handles two modes:
1. Server mode: Starts a Jupyter notebook server (default)
2. Kernel mode: Launches an ipykernel (when called with -m ipykernel_launcher)

Uses notebook 7.x API (JupyterNotebookApp) which is built on jupyter_server.
"""
import sys


def run_kernel():
    """Launch ipykernel when called as a kernel."""
    # Remove the '-m' and 'ipykernel_launcher' from argv
    # The remaining args are what ipykernel expects (e.g., -f connection_file.json)
    idx = sys.argv.index('-m')
    kernel_args = sys.argv[idx + 2:]  # Skip '-m' and 'ipykernel_launcher'
    sys.argv = [sys.argv[0]] + kernel_args

    from ipykernel import kernelapp
    kernelapp.launch_new_instance()


def run_server():
    """Launch Jupyter notebook server."""
    import argparse
    from notebook.app import JupyterNotebookApp

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
        # Disable token authentication for embedded use (local only)
        '--IdentityProvider.token=',
    ]

    if args.notebook_dir:
        argv.append(f'--notebook-dir={args.notebook_dir}')

    if args.config:
        argv.append(f'--config={args.config}')

    JupyterNotebookApp.launch_instance(argv=argv)


def main():
    # Check if we're being called as a kernel launcher
    if '-m' in sys.argv and 'ipykernel_launcher' in sys.argv:
        run_kernel()
    else:
        run_server()


if __name__ == '__main__':
    main()
