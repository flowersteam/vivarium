# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Vivarium - Multi-executable approach with shared dependencies

Builds three executables that share a common dependency folder:
  - vivarium-server (gRPC server)
  - vivarium-interface (Panel web interface, internally calls vivarium-server)
  - vivarium-jupyter (Jupyter notebook server for embedded notebooks)

Uses PyInstaller's MERGE() function to deduplicate shared libraries (JAX, gRPC, etc.)
across executables, significantly reducing total distribution size.

Directory structure after build:
  dist/vivarium/
    vivarium-server        (executable)
    vivarium-interface     (executable)
    vivarium-jupyter       (executable)
    VERSION                (version info, same file as source)
    conf/                  (user-editable Hydra configs)
    notebooks/             (user-editable notebooks)
    _defaults/             (reference copies, updated with app)
      conf/
      notebooks/
    _internal/             (shared Python runtime & libraries)

On first run, if conf/ or notebooks/ don't exist, they are copied from _defaults/.
This allows users to customize while preserving ability to check for upstream changes.

Usage:
    pyinstaller vivarium_multi.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs, copy_metadata

# Get the project root directory
project_root = os.path.abspath(SPECPATH)

# Read version from VERSION file (single source of truth)
with open(os.path.join(project_root, 'VERSION'), 'r') as f:
    VERSION = f.read().strip()

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

# Collect shared data files (libraries)
panel_datas = collect_data_files('panel')
bokeh_datas = collect_data_files('bokeh')

# NOTE: conf and notebooks are NO LONGER included in Analysis datas.
# They are added at the COLLECT stage to place them at the distribution root
# (next to executables) rather than inside _internal.

# Jupyter config for iframe embedding (this stays in _internal, it's not user-editable)
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
    datas=[],  # conf is added at COLLECT stage for user-editability
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
    datas=panel_datas + bokeh_datas + jupyter_config,  # conf/notebooks added at COLLECT stage
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
    datas=jupyter_config + jupyter_pkg_datas,  # conf/notebooks added at COLLECT stage
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
#     VERSION                (version file, same as source)
#     conf/                  (user-editable, copied from _defaults on first run)
#     notebooks/             (user-editable, copied from _defaults on first run)
#     _defaults/             (reference copies, updated with each release)
#       conf/
#       notebooks/
#     _internal/             (shared Python runtime & libraries)
#
# Shared dependencies (JAX, gRPC, NumPy, etc.) appear only once in _internal.

# VERSION file path (single source of truth, included directly in distribution)
version_file_path = os.path.join(project_root, 'VERSION')

# User-editable directories (placed at distribution root)
# These are copied from _defaults/ on first run if they don't exist
user_conf = Tree(os.path.join(project_root, 'conf'), prefix='conf')
user_notebooks = Tree(os.path.join(project_root, 'notebooks'), prefix='notebooks')

# Default/reference copies (for comparison and reset capability)
defaults_conf = Tree(os.path.join(project_root, 'conf'), prefix='_defaults/conf')
defaults_notebooks = Tree(os.path.join(project_root, 'notebooks'), prefix='_defaults/notebooks')

# VERSION file - COLLECT expects 3-tuples: (dest_name, src_path, typecode)
version_data = [('VERSION', version_file_path, 'DATA')]

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
    # User-editable directories (placed in _internal, moved to root in post-processing)
    user_conf,
    user_notebooks,
    # Reference copies in _defaults/
    defaults_conf,
    defaults_notebooks,
    # Version file
    version_data,
    strip=False,
    upx=True,
    name='vivarium',
)

# ============================================================================
# POST-PROCESSING: Move user-editable directories to distribution root
# ============================================================================
# PyInstaller 6+ places all data inside _internal/ by default.
# For user-editable files (conf, notebooks), we want them at the
# distribution root (next to executables) so users can easily access them.
# This section moves these directories after COLLECT completes.

import shutil

dist_dir = os.path.join(project_root, 'dist', 'vivarium')
internal_dir = os.path.join(dist_dir, '_internal')

# Directories/files to move from _internal/ to distribution root
items_to_move = ['conf', 'notebooks', '_defaults', 'VERSION']

for item in items_to_move:
    src = os.path.join(internal_dir, item)
    dst = os.path.join(dist_dir, item)
    if os.path.exists(src):
        # Remove destination if it exists (from previous build)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        # Move from _internal to root
        shutil.move(src, dst)
        print(f"Moved {item} to distribution root")

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
