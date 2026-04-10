"""
Runtime package for Vivarium deployment and process lifecycle.

Provides process management (server, interface, Jupyter, ngrok),
path resolution for frozen/dev modes, and update utilities.
"""

from vivarium.runtime.paths import (
    is_frozen,
    is_macos_app_bundle,
    get_bundle_root,
    get_app_root,
    get_config_dir,
    get_notebooks_dir,
    get_defaults_dir,
    log_runtime_paths,
    get_version,
    initialize_user_data,
    get_jupyter_config_path,
    get_server_command,
    get_interface_command,
    get_jupyter_command,
    DEFAULT_JUPYTER_PORT,
)

from vivarium.runtime._process import (
    kill_port_processes,
    kill_vivarium_processes,
    kill_all_vivarium_processes,
    stop_server_and_interface,
)

from vivarium.runtime._server import (
    wait_for_grpc_server,
    check_server_running,
    start_simulation_server,
    stop_simulation_server,
)

from vivarium.runtime._interface import (
    wait_for_http,
    start_panel_interface,
    stop_panel_interface,
)

from vivarium.runtime._jupyter import (
    register_jupyter_port,
    unregister_jupyter_port,
    get_started_jupyter_ports,
    find_next_available_port,
    start_jupyter_server,
    check_jupyter_running,
    stop_jupyter_server,
)

from vivarium.runtime._ngrok import (
    get_ngrok_token,
    create_ngrok_tunnel,
    close_ngrok_tunnel,
    check_colab_environment,
)
