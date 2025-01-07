This directory contains educational sessions that enable you controlling a simulation from a Notebook controller.

You will need a set of software tools installed on your computer, which are listed below. **If you are unsure about how to install or use them, first ask a professor or another student, we will help you.**
- A Python distribution
    - On Linux it is usually pre-installed
    - On MacOS, the recommended one is [Anaconda](https://www.anaconda.com/)
    - On Windows, we recommend to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), which provide a Linux environment on that platform.
- A virtual environment
    - Either `venv` or `conda`
- [pip](https://pypi.org/project/pip/)
- [git](https://git-scm.com/)
- [Jupyter](https://jupyter.org/)
    - It can usually be installed with `pip`

- If not already done, first install the required software by following the instructions in the main [README](../../).
    - First create a dedicated directory on your computer, e.g. within your `Documents` folder, and execute the intallation instructions from this directory. In the following we will refer to this directory as `<PATH_TO_LOCAL_VIVARIUM_REPO>`.
    - Just follow the steps in the "Installation" section of the README, then come back here.
- Launch `jupyter lab` (or `jupyter notebook`) by opening another terminal (on Windows: open Anaconda Prompt instead) and executing:
```bash
cd <PATH_TO_LOCAL_VIVARIUM_REPO>
source env_vivarium/bin/activate
jupyter notebook
```
- This will open a web page in the browser with a list of files and directories. Go to `notebooks/sessions` and open the practical session you want to do (`session_1.ipynb` if it is the first class).

The rest of the session is described in this newly opened document, please continue from there. 
Here is a quick overview of the available sessions:

- [Session 1](session_1.ipynb): Introduction to basic of the Notebook controller API
- [Session 2](session_2.ipynb): Defining behaviors definition for agents
- [Session 3](session_3.ipynb): Implementing parallel behaviors and add more sensing abilities
- [Session 4](session_4.ipynb): Modulating internal states and sensing other agent's attributes
- [Session 5](session_5_bonus.ipynb): Understanding routines and creating a simple Eco-Evolutionary simulation
- [Session 6](session_6_logging.ipynb): Logging and plotting data
