#!/bin/bash
# =============================================================================
# Vivarium Installation Script (Unix Bootstrap)
# =============================================================================
#
# This script is the entry point for macOS/Linux users. It:
#   1. Detects the platform (Linux, macOS ARM, macOS Intel)
#   2. Validates prerequisites (git, Python)
#   3. Clones the repository
#   4. Invokes install.py for the rest
#
# Usage:
#   # Default (main branch)
#   curl -fsSL https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.sh | bash
#
#   # Specific branch
#   curl -fsSL https://raw.githubusercontent.com/flowersteam/vivarium/BRANCH/install/install.sh | bash -s -- BRANCH
#
# =============================================================================

set -e  # Exit on error

# ============================================================================
# Constants
# ============================================================================

REPO_URL="https://github.com/flowersteam/vivarium.git"
DEFAULT_BRANCH="main"
INSTALL_DIR="vivarium"

# ============================================================================
# Colors
# ============================================================================

# Check if stdout is a terminal
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Output Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Platform Detection
# ============================================================================

detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Linux*)
            PLATFORM="linux"
            PLATFORM_NAME="Linux"
            ;;
        Darwin*)
            if [ "$arch" = "arm64" ]; then
                PLATFORM="macos_arm"
                PLATFORM_NAME="macOS (Apple Silicon)"
            else
                PLATFORM="macos_intel"
                PLATFORM_NAME="macOS (Intel)"
            fi
            ;;
        *)
            print_error "Unsupported operating system: $os"
            print_error "This script is for macOS and Linux. For Windows, use install.ps1"
            exit 1
            ;;
    esac

    echo -e "Detected platform: ${BOLD}$PLATFORM_NAME${NC} ($arch)"
}

# ============================================================================
# Prerequisite Checks
# ============================================================================

check_git() {
    print_step "Checking for git..."

    if command -v git &> /dev/null; then
        print_success "git is installed"
        return 0
    fi

    print_error "git is not installed."
    echo ""
    echo "=== How to Install Git ==="
    echo ""

    case "$PLATFORM" in
        macos_arm|macos_intel)
            echo "For macOS, run:"
            echo "  xcode-select --install"
            echo ""
            echo "Or install via Homebrew:"
            echo "  brew install git"
            ;;
        linux)
            echo "For Ubuntu/Debian:"
            echo "  sudo apt update && sudo apt install git"
            echo ""
            echo "For Fedora:"
            echo "  sudo dnf install git"
            echo ""
            echo "For Arch Linux:"
            echo "  sudo pacman -S git"
            ;;
    esac

    echo ""
    echo "After installing git, run this script again."
    exit 1
}

check_python() {
    print_step "Checking for Python..."

    # Try different Python commands
    local python_cmd=""
    local python_version=""

    for cmd in python3 python; do
        if command -v "$cmd" &> /dev/null; then
            # Get version
            local version_output=$("$cmd" --version 2>&1)
            if [[ "$version_output" =~ Python\ ([0-9]+)\.([0-9]+) ]]; then
                local major="${BASH_REMATCH[1]}"
                local minor="${BASH_REMATCH[2]}"

                # Check if version is supported
                if [ "$major" = "3" ]; then
                    if [ "$minor" = "11" ] || [ "$minor" = "12" ]; then
                        python_cmd="$cmd"
                        python_version="$major.$minor"
                        break
                    fi
                fi
            fi
        fi
    done

    if [ -z "$python_cmd" ]; then
        # No suitable Python found, check what's available
        local found_python=""
        for cmd in python3 python; do
            if command -v "$cmd" &> /dev/null; then
                found_python=$("$cmd" --version 2>&1 || echo "unknown")
                break
            fi
        done

        if [ -n "$found_python" ]; then
            print_error "Found $found_python, but Vivarium requires Python 3.11 or 3.12."
            echo ""
            print_wrong_version_instructions
        else
            print_error "Python 3.11 or 3.12 is required but not found."
            echo ""
            print_python_not_found_instructions
        fi
        exit 1
    fi

    # Intel Mac: strictly require Python 3.11
    if [ "$PLATFORM" = "macos_intel" ]; then
        if [ "$python_version" != "3.11" ]; then
            print_error "Intel Mac requires Python 3.11, but you have Python $python_version."
            echo ""
            echo "Python 3.12 has compatibility issues with some dependencies on Intel Mac."
            echo ""
            echo "If you installed Anaconda, create a Python 3.11 environment:"
            echo ""
            echo "  conda create -n py311 python=3.11"
            echo "  conda activate py311"
            echo ""
            echo "Then run this script again. The installer will create a virtual environment"
            echo "using Python 3.11, which will work independently of conda."
            exit 1
        fi
    fi

    PYTHON_CMD="$python_cmd"
    print_success "Python $python_version found ($python_cmd)"
}

print_wrong_version_instructions() {
    echo "=== How to Fix This ==="
    echo ""
    echo "You have Python installed, but it's the wrong version."
    echo "Create a conda environment with Python 3.11 or 3.12:"
    echo ""

    case "$PLATFORM" in
        macos_intel)
            echo "  conda create -n vivarium python=3.11"
            echo "  conda activate vivarium"
            echo ""
            echo "(Intel Mac requires Python 3.11, not 3.12)"
            ;;
        *)
            echo "  conda create -n vivarium python=3.12"
            echo "  conda activate vivarium"
            ;;
    esac

    echo ""
    echo "Then run this script again."
}

print_python_not_found_instructions() {
    echo "=== How to Install Python ==="
    echo ""

    case "$PLATFORM" in
        macos_arm|macos_intel)
            echo "For macOS, we recommend installing Anaconda:"
            echo ""
            echo "  1. Download from: https://www.anaconda.com/download/success"
            echo "  2. Run the installer"
            echo "  3. Restart your terminal after installation"
            echo ""
            if [ "$PLATFORM" = "macos_intel" ]; then
                echo "  4. Create a Python 3.11 environment (required for Intel Mac):"
                echo "     conda create -n vivarium python=3.11"
                echo "     conda activate vivarium"
                echo ""
            fi
            echo "Alternative: Install via Homebrew:"
            if [ "$PLATFORM" = "macos_intel" ]; then
                echo "  brew install python@3.11"
            else
                echo "  brew install python@3.12"
            fi
            ;;
        linux)
            echo "For Ubuntu/Debian:"
            echo "  sudo apt update && sudo apt install python3.11 python3.11-venv git"
            echo ""
            echo "For Fedora:"
            echo "  sudo dnf install python3.11"
            echo ""
            echo "For other distributions, install Python 3.11 or 3.12 using your package manager"
            echo "or download from https://www.python.org/downloads/"
            ;;
    esac

    echo ""
    echo "After installing Python, run this script again."
}

# ============================================================================
# Repository Clone
# ============================================================================

clone_repository() {
    local branch="$1"

    print_step "Cloning Vivarium repository..."

    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Directory '$INSTALL_DIR' already exists."
        echo "  Checking if it's a valid Vivarium repository..."

        if [ -d "$INSTALL_DIR/.git" ] && [ -f "$INSTALL_DIR/setup.py" ]; then
            echo "  Found existing Vivarium repository."
            echo "  Updating to latest version on branch '$branch'..."
            cd "$INSTALL_DIR"
            git fetch origin
            git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"
            git pull origin "$branch"
            cd ..
            print_success "Repository updated"
            return 0
        else
            print_error "Directory '$INSTALL_DIR' exists but is not a valid Vivarium repository."
            echo "  Please remove or rename it and try again:"
            echo "  rm -rf $INSTALL_DIR"
            exit 1
        fi
    fi

    git clone --branch "$branch" "$REPO_URL" "$INSTALL_DIR"
    print_success "Repository cloned to $INSTALL_DIR"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse arguments
    local branch="${1:-$DEFAULT_BRANCH}"

    print_header "Vivarium Installation"

    echo "This script will install Vivarium, a multi-agent simulation framework."
    echo ""
    echo "Installation details:"
    echo "  - Repository: $REPO_URL"
    echo "  - Branch: $branch"
    echo "  - Install directory: $(pwd)/$INSTALL_DIR"
    echo ""

    # Detect platform
    detect_platform
    echo ""

    # Check prerequisites
    check_git
    check_python
    echo ""

    # Clone repository
    clone_repository "$branch"
    echo ""

    # Run Python installer
    print_step "Running Python installer..."
    cd "$INSTALL_DIR"
    "$PYTHON_CMD" install/install.py --branch "$branch"
}

main "$@"
