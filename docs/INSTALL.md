# Vivarium Installation Guide

This guide will help you install Vivarium, a multi-agent simulation framework. The installation process is designed to be beginner-friendly, even if you have no programming experience.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Platform-Specific Notes](#platform-specific-notes)
5. [Troubleshooting](#troubleshooting)
6. [Installing from a Specific Branch](#installing-from-a-specific-branch)
7. [Updating and Uninstalling](#updating-and-uninstalling)
8. [Verifying Installation](#verifying-installation)

---

## Quick Start

If you already have Python 3.11/3.12 and git installed, use these one-liners:

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.sh | bash
```

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.ps1 -OutFile install.ps1; .\install.ps1
```

If you encounter errors, read the [Prerequisites](#prerequisites) section below.

---

## Prerequisites

### Python Version Requirements

| Platform | Supported Python Versions | Notes |
|----------|---------------------------|-------|
| Ubuntu/Linux | 3.11 or 3.12 | |
| macOS (Apple Silicon) | 3.11 or 3.12 | M1/M2/M3/M4 chips |
| macOS (Intel) | **3.11 only** | See [Intel Mac notes](#intel-mac) |
| Windows | 3.11 or 3.12 | |

### Installing Python (Anaconda - Recommended for macOS/Windows)

Anaconda is a Python distribution that includes many useful tools. We recommend it for macOS and Windows users.

#### Step 1: Download Anaconda

Visit [https://www.anaconda.com/download](https://www.anaconda.com/download) and download the installer for your operating system.

#### Step 2: Run the Installer

**Important options to enable during installation:**

- **macOS/Windows**: Check "Add Anaconda to my PATH environment variable"
- **Windows**: Check "Register Anaconda as my default Python"

These options make Python available from the command line.

#### Step 3: Restart Your Terminal

Close and reopen your terminal (Terminal on macOS, PowerShell on Windows) after installation.

#### Step 4: Verify Installation

```bash
python --version
```

This should show something like `Python 3.12.x` or `Python 3.11.x`.

### Installing Python on Ubuntu/Debian Linux

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv git
```

### Installing Git

Git is required to download Vivarium.

**macOS:**
```bash
xcode-select --install
```

**Windows:**
Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)

During installation, select "Git from the command line and also from 3rd-party software".

**Ubuntu/Debian:**
```bash
sudo apt install git
```

---

## Step-by-Step Installation

### Step 1: Open a Terminal

**macOS:** Open the "Terminal" app (in Applications > Utilities)

**Windows:** Open "PowerShell" (search for it in the Start menu)

**Linux:** Open your terminal emulator

### Step 2: Navigate to Where You Want to Install

Choose where you want Vivarium to be installed. For example:

```bash
cd ~/Documents
```

### Step 3: Run the Installation Script

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.sh | bash
```

**Windows (PowerShell):**
```powershell
iwr -useb https://raw.githubusercontent.com/flowersteam/vivarium/main/install/install.ps1 -OutFile install.ps1; .\install.ps1
```

### Step 4: Wait for Installation

The script will:
1. Check your Python and git installation
2. Clone the Vivarium repository
3. Create a virtual environment
4. Install all dependencies

This may take several minutes depending on your internet connection.

### Step 5: Run Vivarium

After installation completes:

```bash
cd vivarium
./run_vivarium.sh    # macOS/Linux
run_vivarium.bat     # Windows
```

Your web browser will open automatically with the Vivarium interface at http://localhost:5006

---

## Platform-Specific Notes

### Intel Mac

**How to Check if You Have an Intel Mac:**

**Method 1 - GUI:**
1. Click the Apple menu () in the top-left corner
2. Select "About This Mac"
3. Look at the processor/chip:
   - **Intel Mac**: Shows "Processor: Intel Core i5/i7/i9..."
   - **Apple Silicon**: Shows "Chip: Apple M1/M2/M3/M4..."

**Method 2 - Terminal:**
```bash
uname -m
```
- Returns `x86_64` → Intel Mac
- Returns `arm64` → Apple Silicon

**Intel Mac Requirements:**

Intel Macs **require Python 3.11** (not 3.12) due to dependency compatibility issues. If you have Anaconda with Python 3.12, create a Python 3.11 environment:

```bash
conda create -n py311 python=3.11
conda activate py311
```

Then run the installation script. The installer will create a virtual environment using Python 3.11, which will work independently of conda.

### Windows

**PowerShell Execution Policy:**

If you see an error about execution policy, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try the installation again.

**PATH Issues:**

If `python` is not found after installing Anaconda:
1. Restart PowerShell completely (close all windows)
2. If still not working, check that Anaconda was installed with "Add to PATH" enabled
3. You may need to add Python to PATH manually through System Properties > Environment Variables

### Ubuntu/Debian Linux

Make sure you have the venv module:

```bash
sudo apt install python3.11-venv
```

---

## Troubleshooting

### "Python not found" or wrong version

1. Verify Python is installed: `python --version` or `python3 --version`
2. If installed but not found, restart your terminal
3. For Windows: Ensure "Add to PATH" was checked during installation
4. Install Python following the [Prerequisites](#prerequisites) section

### "git not found"

Install git following the instructions in [Prerequisites](#prerequisites).

### Permission denied (Unix)

If you see "Permission denied" errors:

```bash
chmod +x run_vivarium.sh
```

### Installation hangs or times out

This is usually due to slow internet. Try:
1. Wait longer (dependency download can take time)
2. Check your internet connection
3. Run the installation again

### "Address already in use" when starting Vivarium

Another process is using port 5006. Either:
1. Close the other process
2. Wait a few seconds and try again
3. Restart your terminal

### Intel Mac: "grpcio" or JAX errors

Ensure you're using Python 3.11, not 3.12:

```bash
python --version  # Should show 3.11.x
```

If using Anaconda:
```bash
conda create -n py311 python=3.11
conda activate py311
# Run installation again
```

---

## Installing from a Specific Branch

For development or testing purposes, you can install from a specific branch:

### macOS / Linux

```bash
# Replace BRANCH_NAME with the branch you want
curl -fsSL https://raw.githubusercontent.com/flowersteam/vivarium/BRANCH_NAME/install/install.sh | bash -s -- BRANCH_NAME
```

### Windows (PowerShell)

```powershell
# Replace BRANCH_NAME with the branch you want
iwr -useb https://raw.githubusercontent.com/flowersteam/vivarium/BRANCH_NAME/install/install.ps1 -OutFile install.ps1; .\install.ps1 -Branch "BRANCH_NAME"
```

---

## Updating and Uninstalling

### Updating Vivarium

Navigate to the vivarium directory and run:

```bash
cd vivarium
git pull
```

If dependencies changed, re-run the Python installer:

```bash
# Activate the virtual environment first
source venv_vivarium/bin/activate    # macOS/Linux
venv_vivarium\Scripts\activate       # Windows

# Update dependencies
pip install -e .
```

### Uninstalling Vivarium

Simply delete the vivarium directory:

```bash
rm -rf vivarium    # macOS/Linux
Remove-Item -Recurse -Force vivarium    # Windows
```

---

## Verifying Installation

### Quick Test

Start Vivarium:
```bash
cd vivarium
./run_vivarium.sh    # macOS/Linux
run_vivarium.bat     # Windows
```

You should see:
1. "Starting Vivarium server..." message
2. "Starting web interface..." message
3. Your browser opens to http://localhost:5006
4. The Vivarium interface loads with a simulation view

### Python Import Test

```bash
cd vivarium
source venv_vivarium/bin/activate    # macOS/Linux
# or: venv_vivarium\Scripts\activate  # Windows

python -c "import vivarium; print('Vivarium imported successfully!')"
```

### Running Tests

```bash
cd vivarium
source venv_vivarium/bin/activate    # macOS/Linux
# or: venv_vivarium\Scripts\activate  # Windows

pytest tests/ -v --timeout=120
```

---

## Next Steps

After successful installation:

1. **Web Interface Tutorial**: Read `notebooks/tutorials/web_interface_tutorial.md`
2. **Quickstart Notebook**: Open `notebooks/tutorials/quickstart_tutorial.ipynb`
3. **Educational Sessions**: Explore `notebooks/sessions/` for structured learning

Happy simulating!
