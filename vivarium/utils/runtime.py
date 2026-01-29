"""
Runtime environment utilities for development and PyInstaller compatibility.

This module centralizes all logic that differs between development mode
and frozen (PyInstaller) mode, including path resolution and command building.
"""

import os
import sys


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
    Get the root directory for bundled resources.

    Returns:
        - In frozen mode: sys._MEIPASS (PyInstaller's extraction directory)
        - In development: project root directory
    """
    if is_frozen():
        return sys._MEIPASS
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_config_dir() -> str:
    """Get the Hydra configuration directory."""
    return os.path.join(get_bundle_root(), 'conf')


def get_notebooks_dir() -> str:
    """Get the notebooks directory."""
    return os.path.join(get_bundle_root(), 'notebooks')


def get_jupyter_config_path() -> str:
    """Get the path to the Jupyter iframe configuration file."""
    return os.path.join(get_bundle_root(), 'vivarium/interface/jupyter_config_iframe.py')


def _get_frozen_executable_path(name: str) -> str:
    """Get the path to a companion executable in frozen mode."""
    exe_dir = os.path.dirname(sys.executable)

    if is_macos_app_bundle():
        # In .app bundle: all executables are in Contents/MacOS/ together
        exe_path = os.path.join(exe_dir, name)
    else:
        # Folder structure: executables are in sibling directories
        dist_dir = os.path.dirname(exe_dir)
        exe_path = os.path.join(dist_dir, name, name)

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
