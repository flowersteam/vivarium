# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Vivarium - Multi-executable approach with shared dependencies

Builds three executables that share a common dependency folder:
  - vivarium-server (gRPC server)
  - vivarium-interface (Panel web interface, internally calls vivarium-server)
  - vivarium-jupyter (Jupyter notebook server for embedded notebooks)

Uses PyInstaller's MERGE() function to deduplicate shared libraries (JAX, gRPC, etc.)
across executables, significantly reducing total distribution size.

Usage:
    pyinstaller vivarium_multi.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs, copy_metadata

# Get the project root directory
project_root = os.path.abspath(SPECPATH)

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

# Collect shared data files
panel_datas = collect_data_files('panel')
bokeh_datas = collect_data_files('bokeh')
hydra_datas = [(os.path.join(project_root, 'conf'), 'conf')]
notebook_datas = [(os.path.join(project_root, 'notebooks'), 'notebooks')]
# Jupyter config for iframe embedding
jupyter_config = [(os.path.join(project_root, 'vivarium/interface/jupyter_config_iframe.py'), 'vivarium/interface')]

# Collect shared binaries
jax_binaries = collect_dynamic_libs('jax')
jaxlib_binaries = collect_dynamic_libs('jaxlib')
binaries = jax_binaries + jaxlib_binaries

# Shared hidden imports
base_hidden_imports = [
    'jax', 'jax._src', 'jaxlib', 'jax_md',
    'grpc', 'grpcio', 'grpc_health', 'grpc_health.v1',
    'google.protobuf',
    'hydra', 'hydra._internal', 'omegaconf',
    'psutil', 'python_dotenv',
]

# ============================================================================
# SERVER EXECUTABLE
# ============================================================================

server_script = os.path.join(project_root, 'scripts', 'run_server.py')

server_analysis = Analysis(
    [server_script],
    pathex=[project_root],
    binaries=binaries,
    datas=hydra_datas,
    # CRITICAL: Hydra loads classes dynamically via _target_ in YAML configs
    # PyInstaller can't detect these, so we must explicitly collect submodules:
    # - vivarium.environment.components: Server-side JAX components (entities, physics, etc.)
    # Note: This discovers controller.py and interface.py files but doesn't include them
    # because vivarium.environment.__init__.py only imports components (not controllers/interfaces)
    hiddenimports=base_hidden_imports + [
        'vivarium.simulator',
        'vivarium.simulator.grpc_server',
        'vivarium.environment',
        'vivarium.environment.components',
    ] + collect_submodules('vivarium.environment.components'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'panel', 'bokeh', 'notebook', 'jupyter_client',
        'matplotlib', 'matplotlib.pyplot', 'IPython',
        'pandas', 'sklearn', 'tensorflow',
    ],
    noarchive=False,
)

server_pyz = PYZ(server_analysis.pure)

server_exe = EXE(
    server_pyz,
    server_analysis.scripts,
    [],
    exclude_binaries=True,
    name='vivarium-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# ============================================================================
# INTERFACE EXECUTABLE
# ============================================================================

interface_script = os.path.join(project_root, 'scripts', 'run_interface.py')

interface_analysis = Analysis(
    [interface_script],
    pathex=[project_root],
    binaries=binaries,  # Include JAX binaries
    datas=panel_datas + bokeh_datas + hydra_datas + notebook_datas + jupyter_config,
    # CRITICAL: Hydra loads classes dynamically via _target_ and *_cls in YAML configs
    # PyInstaller can't detect these, so we must explicitly collect submodules:
    # - vivarium.controllers.components: Client-side controller APIs (re-exported from vivarium.environment.components)
    # - vivarium.interface.components: UI layer interfaces (re-exported from vivarium.environment.components)
    hiddenimports=base_hidden_imports + [
        'panel', 'bokeh', 'param',
        'vivarium.interface',
        'vivarium.interface.components',
        'vivarium.controllers',
        'vivarium.controllers.components',
    ] + collect_submodules('panel') \
      + collect_submodules('bokeh') \
      + collect_submodules('vivarium.controllers.components') \
      + collect_submodules('vivarium.interface.components'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'matplotlib.pyplot',
        'sklearn', 'tensorflow',
    ],
    noarchive=False,
)

interface_pyz = PYZ(interface_analysis.pure)

interface_exe = EXE(
    interface_pyz,
    interface_analysis.scripts,
    [],
    exclude_binaries=True,
    name='vivarium-interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# ============================================================================
# JUPYTER EXECUTABLE
# ============================================================================

jupyter_script = os.path.join(project_root, 'scripts', 'run_jupyter.py')

# Collect Jupyter package data files (templates, static assets, etc.)
jupyter_pkg_datas = (
    collect_data_files('notebook')
    + collect_data_files('jupyter_server')
    + collect_data_files('jupyter_core')
    + collect_data_files('jupyter_client')
    + collect_data_files('nbformat')
    + collect_data_files('nbconvert')
    + collect_data_files('ipykernel')
    + collect_data_files('jupyter_events')
    + collect_data_files('jsonschema')
    + collect_data_files('rfc3987_syntax')  # Contains .lark grammar files
    + collect_data_files('debugpy')  # Contains _vendored directory needed by ipykernel
    # Include package metadata for entry points (needed for kernel provisioner)
    + copy_metadata('jupyter_client')
    + copy_metadata('jupyter_server')
    + copy_metadata('ipykernel')
)

jupyter_analysis = Analysis(
    [jupyter_script],
    pathex=[project_root],
    binaries=binaries,  # Include JAX binaries for vivarium
    datas=notebook_datas + jupyter_config + jupyter_pkg_datas + hydra_datas,
    hiddenimports=[
        'notebook', 'notebook.app', 'jupyter_server', 'jupyter_client', 'ipykernel',
        'traitlets', 'tornado', 'zmq',
        'ipykernel.datapub', 'ipykernel.comm',
        'jupyter_core', 'nbformat', 'nbconvert',
        'argon2', 'argon2.low_level',  # Password hashing
        'jupyter_server.serverapp',  # Required for notebook 7.x
        # Kernel provisioner (loaded via entry points)
        'jupyter_client.provisioning',
        'jupyter_client.provisioning.factory',
        'jupyter_client.provisioning.local_provisioner',
        # Vivarium package (so notebooks can import it)
        'vivarium',
        'vivarium.controllers',
        'vivarium.simulator',
        'vivarium.simulator.grpc_server',
    ] + collect_submodules('notebook')
      + collect_submodules('jupyter_server')
      + collect_submodules('ipykernel')
      + collect_submodules('jupyter_client')
      + collect_submodules('vivarium'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ============================================================================
# MERGE - Deduplicate shared dependencies across all executables
# ============================================================================
# This significantly reduces the total distribution size by sharing:
# - JAX/JAXlib binaries (~500MB+)
# - gRPC libraries
# - Python standard library
# - NumPy, SciPy, and other scientific packages
#
# After MERGE, each executable references shared files from a common location

MERGE(
    (server_analysis, 'vivarium-server', 'vivarium-server'),
    (interface_analysis, 'vivarium-interface', 'vivarium-interface'),
    (jupyter_analysis, 'vivarium-jupyter', 'vivarium-jupyter'),
)

jupyter_pyz = PYZ(jupyter_analysis.pure)

jupyter_exe = EXE(
    jupyter_pyz,
    jupyter_analysis.scripts,
    [],
    exclude_binaries=True,
    name='vivarium-jupyter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# ============================================================================
# COMBINED COLLECT - Single folder with all executables and shared dependencies
# ============================================================================
# After MERGE, we use a single COLLECT to place all executables together with
# deduplicated binaries and data files. This creates the structure:
#
#   dist/vivarium/
#     vivarium-server        (executable)
#     vivarium-interface     (executable)
#     vivarium-jupyter       (executable)
#     _internal/             (shared Python runtime & libraries)
#
# Shared dependencies (JAX, gRPC, NumPy, etc.) appear only once in _internal.

coll = COLLECT(
    # All three executables
    server_exe,
    interface_exe,
    jupyter_exe,
    # Binaries from all analyses (MERGE has deduplicated these)
    server_analysis.binaries,
    interface_analysis.binaries,
    jupyter_analysis.binaries,
    # Data files from all analyses (MERGE has deduplicated these)
    server_analysis.datas,
    interface_analysis.datas,
    jupyter_analysis.datas,
    strip=False,
    upx=True,
    name='vivarium',
)

# ============================================================================
# NOTE: macOS .app bundle removed for alpha version
# ============================================================================
# For alpha, we distribute raw executables with a launcher script.
# Users double-click the .command script which opens Terminal and runs the app.
# This allows users to see logs and quit cleanly by closing the terminal.
#
# To restore .app bundle in the future, uncomment below:
#
# import sys
# if sys.platform == 'darwin':
#     app = BUNDLE(
#         coll,
#         name='Vivarium.app',
#         icon=None,
#         bundle_identifier='com.vivarium.app',
#         info_plist={
#             'CFBundleName': 'Vivarium',
#             'CFBundleDisplayName': 'Vivarium',
#             'CFBundleVersion': '1.0.0',
#             'CFBundleShortVersionString': '1.0.0',
#             'NSHighResolutionCapable': True,
#             'LSMinimumSystemVersion': '10.15.0',
#         },
#     )
