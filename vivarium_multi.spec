# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Vivarium - Multi-executable approach
Builds two executables:
  - vivarium-server (gRPC server)
  - vivarium-interface (Panel web interface, internally calls vivarium-server)

Usage:
    pyinstaller vivarium_multi.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

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

server_coll = COLLECT(
    server_exe,
    server_analysis.binaries,
    server_analysis.datas,
    strip=False,
    upx=True,
    name='vivarium-server',
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

interface_coll = COLLECT(
    interface_exe,
    interface_analysis.binaries,
    interface_analysis.datas,
    strip=False,
    upx=True,
    name='vivarium-interface',
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
#         interface_coll,
#         server_coll,
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
