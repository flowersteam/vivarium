This directory contains educational sessions that enable you controlling a simulation from a Notebook controller.

You will need a set of software tools installed on your computer, which are listed below. **If you are unsure about how to install or use them, first ask a professor or another student, we will help you.**

- ## 1 - Installation of the required software tools

    - A Python distribution
        - On Linux it is usually pre-installed
        - On MacOS, the recommended one is [Anaconda](https://www.anaconda.com/)
        - On Windows, we recommend to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), which provide a Linux environment on that platform. If you use WSL, you can follow the installation instructions for Linux.
    - A virtual environment
        - Either `venv` or `conda` (we recommend `venv`, which usually comes pre-installed with Python)
    - [pip](https://pypi.org/project/pip/)
    - [git](https://git-scm.com/)


    - First create a dedicated directory on your computer, e.g. within your `Documents` folder, and execute the intallation instructions from this directory. In the following we will refer to this directory as `<PATH_TO_LOCAL_VIVARIUM_REPO>`.
    - Just follow the installation steps below.

- ## 2 - Installation of the Vivarium project

    - ### 2.1 - Automatic installation (not available for Windows)

    - If you work from an UPF computer with Ubuntu (Linux), or from a machine with any Unix distribution (Linux or Mac OS), you can do an automatic installation of the project. To do so open a terminal, navigate to the vivarium directory that you created and execute the following commands:

        ```bash
        # update the package list (you might need to enter your password)
        sudo apt update
        # install wget if you don't have it
        sudo apt install wget
        # download the automatic installation script
        wget https://raw.githubusercontent.com/flowersteam/vivarium/refs/heads/main/linux_install.sh
        # make it executable
        chmod u+x linux_install.sh
        # run the script
        ./install.sh
        ```

    - This will install the project and all its dependencies. You can then skip to the next sub-sections and directly start the educational sessions (see how to do it in part 3 below).

    - ### 2.1 - Manual installation (for all platforms)

    - #### 1- Clone the repository:

        Before following the next instructions, make sure you have Python installed with a version between 3.10 and 3.12. 

        ```bash
        # first clone the repository (copy and execute the right command for your case)
        git clone https://github.com/flowersteam/vivarium.git #(if you don't have a GitHub account)
        git clone git@github.com:flowersteam/vivarium.git #(if you have a GitHub account and SSH keys set up)

        # then go to the repository directory
        cd vivarium/
        ```
    - #### 2- Create and activate a virtual environment:

        For Linux users:

        ```bash
        # create a virtual environment
        python3 -m venv env_vivarium

        # if the above command doesn't work and you are asked to install the `venv` module, execute this:
        sudo apt install -y python3-venv
        # (then run the first command again to create the virtual environment)

        # activate the virtual environment
        source env_vivarium/bin/activate #(for Linux users)
        ```

        For Windows users (in PowerShell):
        ```bash
        # create a virtual environment
        python -m venv env_vivarium

        # activate the virtual environment
        env_vivarium\Scripts\Activate.ps1 #(for Windows users)
        ```

    - #### 3- Install the dependencies:

        You should now have an indication in your terminal that you are in the virtual environment, e.g. `(env_vivarium)`. You can now install the dependencies inside:

        ```bash
        pip install -r requirements.txt
        pip install -e . 
        ```

        Now you are ready to start the Jupyter Notebook server and open the educational sessions.

- ## 3 - Use the project

    - From now on you will start every session by launching `jupyter lab` (or `jupyter notebook`). To do so, open another terminal (on Windows: use PowerShell), navigate to the repository directory, activate the virtual environment, and start the Jupyter Notebook server:

        ```bash
        # go to the repository directory
        cd <PATH_TO_LOCAL_VIVARIUM_REPO>

        # download the latest changes from the repository if there are any
        git pull

        # activate the virtual environment to have access to the installed dependencies
        source env_vivarium/bin/activate #(for Linux users)
        env_vivarium\Scripts\Activate.ps1 #(for Windows users)

        # start the Jupyter Notebook server
        jupyter notebook
        ```
    - This will open a web page in the browser with a list of files and directories. Go to `notebooks/sessions` and open the practical session you want to do (`session_1.ipynb` if it is the first class).

    - if you are a Windows user without WSL, you will also need to start the server and the interface manually from command line (it will be mentionned in the notebook). To do so, open a new terminal (PowerShell) and navigate to the repository directory, activate the virtual environment, and start them with the following commands:

        ```bash
        .\start_all.bat session_1 # for the first session, change the number for the desired session
        ```

The rest of the session is described in this newly opened document, please continue from there. 
Here is a quick overview of the available sessions:

- [Session 1](session_1.ipynb): Introduction to basic of the Notebook controller API
- [Session 2](session_2.ipynb): Defining behaviors definition for agents
- [Session 3](session_3.ipynb): Implementing parallel behaviors and add more sensing abilities
- [Session 4](session_4.ipynb): Modulating internal states and sensing other agent's attributes
- [Session 5](session_5_logging.ipynb): Logging and plotting data
- [Session 6](session_6_bonus.ipynb): Understanding routines and creating a simple Eco-Evolutionary simulation

If you have to configure your own simulation for a project, you can have a look at the [custom scene creation tutorial (student version)](./create_custom_scene_tutorial_simple.md).