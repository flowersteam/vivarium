# =============================================================================
# Vivarium Installation Script (Windows Bootstrap)
# =============================================================================
#
# This script is the entry point for Windows users. It:
#   1. Validates prerequisites (git, Python)
#   2. Clones the repository
#   3. Invokes install.py for the rest
#
# Usage:
#   # Download and run (main branch)
#   iwr -useb https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.ps1 -OutFile install.ps1; powershell -ExecutionPolicy Bypass -File install.ps1
#
#   # Specific branch
#   iwr -useb https://raw.githubusercontent.com/flowersteam/vivarium/BRANCH/install/install.ps1 -OutFile install.ps1; powershell -ExecutionPolicy Bypass -File install.ps1 -Branch BRANCH
#
# =============================================================================

param(
    [string]$Branch = "main"
)

# Strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================================
# Constants
# ============================================================================

$REPO_URL = "https://github.com/flowersteam/vivarium.git"
$INSTALL_DIR = "vivarium"

# ============================================================================
# Output Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# ============================================================================
# Prerequisite Checks
# ============================================================================

function Test-Git {
    Write-Step "Checking for git..."

    try {
        $null = Get-Command git -ErrorAction Stop
        Write-Success "git is installed"
        return $true
    }
    catch {
        Write-Error "git is not installed."
        Write-Host ""
        Write-Host "=== How to Install Git ===" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Download from: https://git-scm.com/download/win"
        Write-Host ""
        Write-Host "During installation, keep the default options and make sure to enable:"
        Write-Host '  "Git from the command line and also from 3rd-party software"'
        Write-Host ""
        Write-Host "After installing git, restart PowerShell and run this script again."
        return $false
    }
}

function Test-Python {
    Write-Step "Checking for Python..."

    # Try different Python commands
    $pythonCmd = $null
    $pythonVersion = $null

    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $versionOutput = & $cmd --version 2>&1
            if ($versionOutput -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]

                if ($major -eq 3 -and ($minor -eq 11 -or $minor -eq 12)) {
                    $pythonCmd = $cmd
                    $pythonVersion = "$major.$minor"
                    break
                }
            }
        }
        catch {
            # Command not found, continue
        }
    }

    if (-not $pythonCmd) {
        # Check what Python is available
        $foundPython = $null
        foreach ($cmd in @("python", "python3", "py")) {
            try {
                $foundPython = & $cmd --version 2>&1
                break
            }
            catch {}
        }

        if ($foundPython) {
            Write-Error "Found $foundPython, but Vivarium requires Python 3.11 or 3.12."
            Write-Host ""
            Write-WrongVersionInstructions
        }
        else {
            Write-Error "Python 3.11 or 3.12 is required but not found."
            Write-Host ""
            Write-PythonNotFoundInstructions
        }
        return $false
    }

    $script:PYTHON_CMD = $pythonCmd
    Write-Success "Python $pythonVersion found ($pythonCmd)"
    return $true
}

function Write-WrongVersionInstructions {
    Write-Host "=== How to Fix This ===" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You have Python installed, but it's the wrong version."
    Write-Host "Create a conda environment with Python 3.11 or 3.12:"
    Write-Host ""
    Write-Host "  conda create -n vivarium python=3.11" -ForegroundColor Green
    Write-Host "  conda activate vivarium" -ForegroundColor Green
    Write-Host ""
    Write-Host "Then run this script again."
    Write-Host ""
}

function Write-PythonNotFoundInstructions {
    Write-Host "=== How to Fix This ===" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "If you already have Anaconda installed:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. Press the Windows key and search for 'Anaconda Prompt'"
    Write-Host "  2. Open it"
    Write-Host "  3. Run: " -NoNewline
    Write-Host "conda init powershell" -ForegroundColor Green
    Write-Host "  4. Close the Anaconda Prompt"
    Write-Host "  5. Open a new PowerShell window and run this script again"
    Write-Host ""
    Write-Host "If you don't have Python installed:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Option A - Anaconda (recommended):"
    Write-Host "    1. Download from: https://www.anaconda.com/download/success"
    Write-Host "    2. Run the installer"
    Write-Host "    3. Follow the steps above to initialize conda"
    Write-Host ""
    Write-Host "  Option B - Python from python.org:"
    Write-Host "    1. Download from: https://www.python.org/downloads/"
    Write-Host "    2. During installation, check 'Add Python to PATH'"
    Write-Host "    3. Restart PowerShell and run this script again"
    Write-Host ""
}

# ============================================================================
# Repository Clone
# ============================================================================

function Install-Repository {
    param([string]$BranchName)

    Write-Step "Cloning Vivarium repository..."

    if (Test-Path $INSTALL_DIR) {
        Write-Warning "Directory '$INSTALL_DIR' already exists."
        Write-Host "  Checking if it's a valid Vivarium repository..."

        if ((Test-Path "$INSTALL_DIR\.git") -and (Test-Path "$INSTALL_DIR\setup.py")) {
            Write-Host "  Found existing Vivarium repository."
            Write-Host "  Updating to latest version on branch '$BranchName'..."

            Push-Location $INSTALL_DIR
            try {
                git fetch origin
                # Try to checkout the branch
                $null = git checkout $BranchName 2>&1
                if ($LASTEXITCODE -ne 0) {
                    $null = git checkout -b $BranchName "origin/$BranchName" 2>&1
                }
                git pull origin $BranchName
                Write-Success "Repository updated"
            }
            finally {
                Pop-Location
            }
            return $true
        }
        else {
            Write-Error "Directory '$INSTALL_DIR' exists but is not a valid Vivarium repository."
            Write-Host "  Please remove or rename it and try again:"
            Write-Host "  Remove-Item -Recurse -Force $INSTALL_DIR"
            return $false
        }
    }

    git clone --branch $BranchName $REPO_URL $INSTALL_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to clone repository"
        return $false
    }

    Write-Success "Repository cloned to $INSTALL_DIR"
    return $true
}

# ============================================================================
# Main
# ============================================================================

function Main {
    Write-Header "Vivarium Installation"

    Write-Host "This script will install Vivarium, a multi-agent simulation framework."
    Write-Host ""
    Write-Host "Installation details:"
    Write-Host "  - Repository: $REPO_URL"
    Write-Host "  - Branch: $Branch"
    Write-Host "  - Install directory: $(Get-Location)\$INSTALL_DIR"
    Write-Host ""
    Write-Host "Detected platform: Windows"
    Write-Host ""

    # Check prerequisites
    if (-not (Test-Git)) {
        exit 1
    }

    if (-not (Test-Python)) {
        exit 1
    }
    Write-Host ""

    # Clone repository
    if (-not (Install-Repository -BranchName $Branch)) {
        exit 1
    }
    Write-Host ""

    # Run Python installer
    Write-Step "Running Python installer..."
    Push-Location $INSTALL_DIR
    try {
        & $PYTHON_CMD install/install.py --branch $Branch
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Installation failed"
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}

# Run main
Main
