#!/usr/bin/env python3
"""
Vivarium Installation Script

Cross-platform Python installer that:
1. Creates a virtual environment
2. Installs dependencies via pip
3. Patches jax-md on Intel Mac
4. Generates convenience run scripts

Usage:
    python install/install.py [--ci] [--branch BRANCH]

Options:
    --ci      Non-interactive CI mode (no prompts, fail on error)
    --branch  Branch name (used for display purposes only)
"""

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

VENV_NAME = "venv_vivarium"
SUPPORTED_PYTHON_VERSIONS = [(3, 11), (3, 12)]
INTEL_MAC_PYTHON_VERSION = (3, 11)


# ============================================================================
# Color Output
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY or Windows without ANSI support)."""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''


def supports_color():
    """Check if the terminal supports color output."""
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    if platform.system() == 'Windows':
        # Windows 10+ supports ANSI colors in cmd.exe
        return os.environ.get('ANSICON') or 'TERM' in os.environ
    return True


def print_header(msg):
    """Print a header message."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def print_step(step_num, total, msg):
    """Print a step progress message."""
    print(f"{Colors.BLUE}[{step_num}/{total}]{Colors.ENDC} {msg}")


def print_success(msg):
    """Print a success message."""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {msg}")


def print_warning(msg):
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {msg}")


def print_error(msg):
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {msg}")


# ============================================================================
# Platform Detection
# ============================================================================

def get_platform_info():
    """
    Detect the current platform.

    Returns:
        dict with keys: system, machine, is_intel_mac
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    is_intel_mac = (system == 'darwin' and machine == 'x86_64')

    return {
        'system': system,
        'machine': machine,
        'is_intel_mac': is_intel_mac,
        'is_macos': system == 'darwin',
        'is_linux': system == 'linux',
        'is_windows': system == 'windows',
    }


def get_python_version():
    """Get the current Python version as a tuple."""
    return (sys.version_info.major, sys.version_info.minor)


def validate_python_version(platform_info, ci_mode=False):
    """
    Validate that the Python version is compatible.

    Args:
        platform_info: Platform information dict
        ci_mode: If True, exit on error instead of providing guidance

    Returns:
        True if valid, exits otherwise
    """
    version = get_python_version()
    version_str = f"{version[0]}.{version[1]}"

    # Intel Mac requires Python 3.11 only
    if platform_info['is_intel_mac']:
        if version != INTEL_MAC_PYTHON_VERSION:
            print_error(f"Intel Mac requires Python {INTEL_MAC_PYTHON_VERSION[0]}.{INTEL_MAC_PYTHON_VERSION[1]}, but you have Python {version_str}.")
            print()
            if not ci_mode:
                print("Python 3.12 has compatibility issues with some dependencies on Intel Mac.")
                print("If you installed Anaconda, create a Python 3.11 environment:")
                print()
                print("  conda create -n py311 python=3.11")
                print("  conda activate py311")
                print()
                print("Then run this script again.")
            sys.exit(1)
    else:
        # Other platforms: Python 3.11 or 3.12
        if version not in SUPPORTED_PYTHON_VERSIONS:
            supported = ', '.join(f"{v[0]}.{v[1]}" for v in SUPPORTED_PYTHON_VERSIONS)
            print_error(f"Python {supported} required, but you have Python {version_str}.")
            if not ci_mode:
                print()
                print("Please install a supported Python version and try again.")
            sys.exit(1)

    print_success(f"Python {version_str} detected")
    return True


# ============================================================================
# Virtual Environment
# ============================================================================

def get_venv_path(base_path):
    """Get the path to the virtual environment."""
    return base_path / VENV_NAME


def get_venv_python(venv_path, platform_info):
    """Get the path to the Python executable in the venv."""
    if platform_info['is_windows']:
        return venv_path / 'Scripts' / 'python.exe'
    return venv_path / 'bin' / 'python'


def get_venv_pip(venv_path, platform_info):
    """Get the path to the pip executable in the venv."""
    if platform_info['is_windows']:
        return venv_path / 'Scripts' / 'pip.exe'
    return venv_path / 'bin' / 'pip'


def create_venv(venv_path, ci_mode=False):
    """
    Create a virtual environment.

    Args:
        venv_path: Path to create the venv at
        ci_mode: If True, overwrite existing venv without prompting

    Returns:
        True if created successfully
    """
    if venv_path.exists():
        if ci_mode:
            print(f"Removing existing virtual environment at {venv_path}")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_warning(f"Virtual environment already exists at {venv_path}")
            print("  It will be reused. Delete it manually if you want a fresh install.")
            return True

    print(f"Creating virtual environment at {venv_path}...")

    try:
        venv.create(venv_path, with_pip=True)
        print_success("Virtual environment created")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        sys.exit(1)


# ============================================================================
# Dependency Installation
# ============================================================================

def run_command(cmd, description, cwd=None, env=None):
    """
    Run a command and handle errors.

    Args:
        cmd: Command list to run
        description: Human-readable description
        cwd: Working directory
        env: Environment variables

    Returns:
        True if successful
    """
    print(f"  Running: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print_error(f"{description} failed")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False

        return True
    except Exception as e:
        print_error(f"{description} failed: {e}")
        return False


def upgrade_pip(venv_python):
    """Upgrade pip in the virtual environment."""
    print("Upgrading pip...")
    cmd = [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip']
    if not run_command(cmd, "Pip upgrade"):
        print_warning("Pip upgrade failed, continuing anyway...")
    else:
        print_success("Pip upgraded")


def install_dependencies(venv_pip, base_path):
    """
    Install dependencies using pip install -e .

    Args:
        venv_pip: Path to pip in the venv
        base_path: Path to the repository root

    Returns:
        True if successful
    """
    print("Installing dependencies (this may take a few minutes)...")
    cmd = [str(venv_pip), 'install', '-e', '.']

    # Run with real-time output for better user experience
    try:
        process = subprocess.Popen(
            cmd,
            cwd=base_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            # Show progress indicators but not every line
            line = line.strip()
            if line and (
                line.startswith('Collecting') or
                line.startswith('Installing') or
                line.startswith('Successfully') or
                'error' in line.lower()
            ):
                print(f"  {line}")

        process.wait()

        if process.returncode != 0:
            print_error("Dependency installation failed")
            return False

        print_success("Dependencies installed")
        return True

    except Exception as e:
        print_error(f"Dependency installation failed: {e}")
        return False


def patch_jax_md(venv_python, base_path):
    """
    Run the jax-md patch script for Intel Mac.

    Args:
        venv_python: Path to Python in the venv
        base_path: Path to the repository root

    Returns:
        True if successful
    """
    print("Patching jax-md for Intel Mac compatibility...")
    patch_script = base_path / 'scripts' / 'patch_jax_md.py'

    if not patch_script.exists():
        print_error(f"Patch script not found: {patch_script}")
        return False

    cmd = [str(venv_python), str(patch_script)]
    if run_command(cmd, "jax-md patch", cwd=base_path):
        print_success("jax-md patched")
        return True
    return False


# ============================================================================
# Convenience Scripts
# ============================================================================

def generate_run_scripts(base_path, platform_info):
    """
    Generate convenience scripts for running Vivarium.

    Args:
        base_path: Path to the repository root
        platform_info: Platform information dict
    """
    print("Generating convenience run scripts...")

    # Unix script
    unix_script = base_path / 'run_vivarium.sh'
    unix_content = '''#!/bin/bash
# Convenience script to run Vivarium
# Generated by install/install.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv_vivarium/bin/activate

# Run Vivarium
python scripts/run_vivarium.py "$@"
'''

    # Windows script
    windows_script = base_path / 'run_vivarium.bat'
    windows_content = '''@echo off
REM Convenience script to run Vivarium
REM Generated by install/install.py

cd /d "%~dp0"

REM Activate virtual environment
call venv_vivarium\\Scripts\\activate.bat

REM Run Vivarium
python scripts\\run_vivarium.py %*
'''

    # Write Unix script
    try:
        unix_script.write_text(unix_content)
        unix_script.chmod(0o755)
        print_success(f"Created {unix_script.name}")
    except Exception as e:
        print_warning(f"Could not create {unix_script.name}: {e}")

    # Write Windows script
    try:
        windows_script.write_text(windows_content)
        print_success(f"Created {windows_script.name}")
    except Exception as e:
        print_warning(f"Could not create {windows_script.name}: {e}")


# ============================================================================
# Main Installation
# ============================================================================

def print_success_message(base_path, platform_info, branch=None):
    """Print the success message with next steps."""
    print_header("Installation Complete!")

    if platform_info['is_windows']:
        run_cmd = ".\\run_vivarium.bat"
        activate_cmd = r"venv_vivarium\Scripts\activate"
    else:
        run_cmd = "./run_vivarium.sh"
        activate_cmd = "source venv_vivarium/bin/activate"

    print(f"{Colors.GREEN}Vivarium has been installed successfully!{Colors.ENDC}")
    print()
    print("To start Vivarium:")
    print(f"  cd {base_path}")
    print(f"  {run_cmd}")
    print()
    print("Or activate the environment manually:")
    print(f"  cd {base_path}")
    print(f"  {activate_cmd}")
    print("  python scripts/run_vivarium.py")
    print()
    print("The web interface will open in your browser at http://localhost:5006")
    print()

    if branch and branch != 'main':
        print(f"{Colors.CYAN}Note: You installed from branch '{branch}'{Colors.ENDC}")
        print()

    print("For tutorials, see:")
    print("  - notebooks/tutorials/quickstart_tutorial.ipynb")
    print("  - notebooks/tutorials/web_interface_tutorial.md")
    print()
    print(f"{Colors.CYAN}Happy simulating!{Colors.ENDC}")


def main():
    """Main installation entry point."""
    parser = argparse.ArgumentParser(
        description="Install Vivarium multi-agent simulation framework"
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: non-interactive, fail on errors'
    )
    parser.add_argument(
        '--branch',
        type=str,
        default=None,
        help='Branch name (for display purposes)'
    )
    args = parser.parse_args()

    # Determine base path (assume we're running from inside the repo)
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent

    # Verify we're in the right directory
    if not (base_path / 'setup.py').exists():
        print_error("Cannot find setup.py. Make sure you're running from the vivarium repository.")
        sys.exit(1)

    # Setup colors
    if not supports_color():
        Colors.disable()

    # Print header
    print_header("Vivarium Installation")
    print(f"Installing from: {base_path}")
    if args.branch:
        print(f"Branch: {args.branch}")
    print()

    # Detect platform
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']} ({platform_info['machine']})")
    if platform_info['is_intel_mac']:
        print(f"{Colors.YELLOW}Note: Intel Mac detected - some special handling will be applied{Colors.ENDC}")
    print()

    total_steps = 5 if platform_info['is_intel_mac'] else 4
    current_step = 0

    # Step 1: Validate Python version
    current_step += 1
    print_step(current_step, total_steps, "Validating Python version...")
    validate_python_version(platform_info, ci_mode=args.ci)

    # Step 2: Create virtual environment
    current_step += 1
    print_step(current_step, total_steps, "Creating virtual environment...")
    venv_path = get_venv_path(base_path)
    create_venv(venv_path, ci_mode=args.ci)

    venv_python = get_venv_python(venv_path, platform_info)
    venv_pip = get_venv_pip(venv_path, platform_info)

    # Step 3: Upgrade pip and install dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing dependencies...")
    upgrade_pip(venv_python)
    if not install_dependencies(venv_pip, base_path):
        print_error("Installation failed. Please check the errors above.")
        sys.exit(1)

    # Step 4: Patch jax-md on Intel Mac
    if platform_info['is_intel_mac']:
        current_step += 1
        print_step(current_step, total_steps, "Applying Intel Mac patches...")
        if not patch_jax_md(venv_python, base_path):
            print_error("jax-md patching failed. The installation may not work correctly.")
            if args.ci:
                sys.exit(1)

    # Step 5: Generate convenience scripts
    current_step += 1
    print_step(current_step, total_steps, "Generating convenience scripts...")
    generate_run_scripts(base_path, platform_info)

    # Print success message
    print_success_message(base_path, platform_info, branch=args.branch)

    return 0


if __name__ == '__main__':
    sys.exit(main())
