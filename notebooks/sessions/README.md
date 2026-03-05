This document contains instructions on how to install Vivarium, the software we will use for the practical sessions.

# Install Vivarium
## Download the version corresponding to your operating system:

You will find the last version of Vivarium at https://github.com/flowersteam/vivarium/releases/latest

You need to download the version corresponding to your operating system:

- **For Linux (Ubuntu):**  Download `vivarium-linux-x64.tar.gz`
- **For MacOS (Apple Silicon M1/.../M4):** Download `vivarium-macos-arm64.tar.gz`
- **For MacOS (Intel)**: Download `vivarium-macos-x64.tar.gz`
- **For Windows**: Download `vivarium-windows-x64.zip`

If you don't know if you are on a Mac with Apple Silicon vs. Intel:
- Click the Apple menu in the top-left corner of your screen
- Select About This Mac
- Look for the Chip or Processor information:
    - If you are on Apple Silicon it will say "Apple M1", "Apple M2", "Apple M3", or "Apple M4"
    - If you are on Intel it will say something like "Intel Core i5" or "Intel Core i7"

## Extract the downloaded archive

Place the downloaded file in a folder where you can easily find it again, e.g. your `Documents` folder.

Then extract the archive:
- On Windows: right-click on the file, then "Extract all".
- On MacOs and Linux Ubuntu: double-click on the file. 

## First launch

In the extracted folder, you will find a file called `Start-Vivarium`, which will start the software. 

- **On Linux Ubuntu:** right-click on the file and choose `Run as a program`.
- **On MacOS:** double-click on the file. It will most likely refuse to open it because it was downloaded from internet and not from the AppStore. To allow opening it, you will have to go to `Settings -> Privacy and Security`. Near the bottom of these settings click on "Open Anyway". Then click on "Open Anyway" again (it will ask for your password since this is a security setting).
- **On Windows:** Double-click on the file. It will show a security warning because the app is downloaded from internet, choose "Run".

A terminal window will open. Wait a bit for the installation to proceed. Once completed it will open a tab in your browser with the Vivarium interface.

On Windows, it might ask you to enable network connections at some points. Always choose "Allow".

# Open a session

Once the Vivarium interface appears in your browser, you will be able to select a scene. The practical sessions correspond to scene prefixed with `Sessions:...`. Select the first session, i.e. `Sessions: session_1` and click "Start Simulation". Wait a bit for the session to open.

Then click the button `Start Jupyter Server`, then on `Open session_1.ipynb`. A Jupyter Notebook will open. This is your starting point, all the instructions for the session are in this notebook. Read the rest of the instructions below, then start following the instructions in the notebook. 

# Quit Vivarium

Once you have completed your session, or if you want to stop and restart later, first **save your notebook**. At the top of the notebook there is a small "floppy disk" icon which will save the notebook if you click it. Or you can do File -> Save Notebook (in the File menu of the notebook panel in the webpage, **not** the File menu of your browser). It is recommended to regularly save your notebook to avoid losing your work if the something goes wrong. 

Once your notebook is saved, you can quit Vivarium by pressing `Ctrl-C` on the terminal window that spawned when you first opened it (on some operating systems you might need to press `Ctrl-C` twice and to confirm you really want to quit by entering `Y` in the terminal).

When you want to continue on your sessions, you can reopen it [as explained above](#first-launch) and open the session you want to work on.

# Updating Vivarium

If a new version of Vivarium becomes available it will notify at start-up and propose to update the software. We recommend you to do the update in this case. 