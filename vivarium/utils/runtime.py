"""
Runtime environment utilities for development and PyInstaller compatibility.

This module centralizes all logic that differs between development mode
and frozen (PyInstaller) mode, including path resolution and command building.
It also handles first-run initialization for frozen builds.

Update-related functionality is in vivarium.utils.updater.
"""

import os
import sys
import shutil
import logging


lg = logging.getLogger(__name__)


def is_frozen() -> bool:
    """Check if running as a PyInstaller bundle."""
    return getattr(sys, 'frozen', False)


def is_macos_app_bundle() -> bool:
    """Check if running from a macOS .app bundle."""
    return (
        is_frozen()
        and sys.platform == 'darwin'
        and '.app/Contents/MacOS' in sys.executable
    )


def get_bundle_root() -> str:
    """
    Get the root directory for bundled resources (internal/read-only).

    Returns:
        - In frozen mode: sys._MEIPASS (PyInstaller's extraction directory)
        - In development: project root directory
    """
    if is_frozen():
        return sys._MEIPASS
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_app_root() -> str:
    """
    Get the application root directory (where user-editable files are located).

    Returns:
        - In frozen mode: directory containing the executable (and conf/, notebooks/)
        - In development: project root directory
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_config_dir() -> str:
    """Get the Hydra configuration directory (user-editable in frozen mode)."""
    if is_frozen():
        return os.path.join(get_app_root(), 'conf')
    return os.path.join(get_bundle_root(), 'conf')


def get_notebooks_dir() -> str:
    """Get the notebooks directory (user-editable in frozen mode)."""
    if is_frozen():
        return os.path.join(get_app_root(), 'notebooks')
    return os.path.join(get_bundle_root(), 'notebooks')


def get_defaults_dir() -> str:
    """
    Get the defaults directory containing reference copies of conf and notebooks.

    Only meaningful in frozen mode. In development, returns the bundle root
    since conf/ and notebooks/ are already at the project root.
    """
    if is_frozen():
        return os.path.join(get_app_root(), '_defaults')
    return get_bundle_root()


def log_runtime_paths() -> None:
    """Log all runtime paths for debugging purposes."""
    mode = "frozen (PyInstaller)" if is_frozen() else "development"
    lg.info(f"Runtime mode: {mode}")
    lg.info(f"  App root:       {get_app_root()}")
    lg.info(f"  Bundle root:    {get_bundle_root()}")
    lg.info(f"  Config dir:     {get_config_dir()}")
    lg.info(f"  Notebooks dir:  {get_notebooks_dir()}")
    if is_frozen():
        lg.info(f"  Defaults dir:   {get_defaults_dir()}")


def get_version() -> str:
    """
    Get the current application version from the VERSION file.

    Works in both frozen and development modes by using get_app_root().

    Raises:
        FileNotFoundError: If the VERSION file is missing (indicates broken installation).
    """
    version_file = os.path.join(get_app_root(), 'VERSION')
    with open(version_file, 'r') as f:
        return f.read().strip()


def initialize_user_data() -> bool:
    """
    Initialize user data directories on first run (frozen mode only).

    Copies conf/ and notebooks/ from _defaults/ to the distribution root
    if they don't already exist.

    Returns:
        True if initialization was performed, False if already initialized
        or not in frozen mode.
    """
    if not is_frozen():
        return False

    dist_dir = get_app_root()
    defaults_dir = get_defaults_dir()

    initialized = False

    for folder in ['conf', 'notebooks']:
        user_folder = os.path.join(dist_dir, folder)
        default_folder = os.path.join(defaults_dir, folder)

        if not os.path.exists(user_folder) and os.path.exists(default_folder):
            lg.info(f"First run: copying {folder}/ from defaults...")
            try:
                shutil.copytree(default_folder, user_folder)
                lg.info(f"Successfully initialized {folder}/")
                initialized = True
            except Exception as e:
                lg.error(f"Failed to initialize {folder}/: {e}")

    return initialized


def get_jupyter_config_path() -> str:
    """Get the path to the Jupyter iframe configuration file."""
    return os.path.join(get_bundle_root(), 'vivarium/interface/jupyter_config_iframe.py')


def _get_frozen_executable_path(name: str) -> str:
    """Get the path to a companion executable in frozen mode.

    All executables (vivarium-server, vivarium-interface, vivarium-jupyter)
    are in the same directory, sharing a common _internal folder.
    """
    exe_dir = os.path.dirname(sys.executable)
    exe_path = os.path.join(exe_dir, name)

    # On Windows, add .exe extension
    if sys.platform == 'win32':
        exe_path += '.exe'

    return exe_path


def get_server_command(cmd_args: list) -> list:
    """
    Get the command to start the simulation server.

    Args:
        cmd_args: Command line arguments to pass to the server

    Returns:
        Full command list ready for subprocess
    """
    if is_frozen():
        server_exe = _get_frozen_executable_path('vivarium-server')
        return [server_exe, *cmd_args]
    else:
        server_script = os.path.join(get_bundle_root(), 'scripts/run_server.py')
        return [sys.executable, server_script, *cmd_args]


def get_interface_command(allow_external_origins: bool = False) -> list:
    """
    Get the command to start the web interface.

    Args:
        allow_external_origins: Whether to allow websocket connections from external origins

    Returns:
        Full command list ready for subprocess
    """
    if is_frozen():
        interface_exe = _get_frozen_executable_path('vivarium-interface')
        return [interface_exe]
    else:
        interface_script = os.path.join(get_bundle_root(), 'scripts/run_interface.py')
        command = [sys.executable, interface_script, "--dont-open-browser"]
        if allow_external_origins:
            command.append("--allow-external-origins")
        return command


def get_jupyter_command(port: int = 8889, notebook_dir: str = None, config_path: str = None) -> list:
    """
    Get the command to start the Jupyter server.

    In frozen mode: returns path to vivarium-jupyter executable
    In dev mode: returns 'jupyter notebook' CLI command

    Args:
        port: Port to run Jupyter on
        notebook_dir: Directory to start Jupyter in
        config_path: Path to Jupyter config file

    Returns:
        Full command list ready for subprocess
    """
    if is_frozen():
        cmd = [_get_frozen_executable_path('vivarium-jupyter')]
        cmd.extend(['--port', str(port)])
        if notebook_dir:
            cmd.extend(['--notebook-dir', str(notebook_dir)])
        if config_path:
            cmd.extend(['--config', str(config_path)])
    else:
        cmd = [
            'jupyter', 'notebook',
            f'--port={port}',
            '--no-browser',
        ]
        if notebook_dir:
            cmd.append(f'--notebook-dir={notebook_dir}')
        if config_path:
            cmd.append(f'--config={config_path}')

    return cmd
