This directory contains educational sessions that enable you controlling a simulation from a Notebook controller.

- If not already done, first install the required software by following the instructions in the main [README](../../).
    - For MacOS and Windows: If Python and/or Jupyter Notebook is not installed on your computer, first [install Anaconda](https://www.anaconda.com/).
- Launch `jupyter notebook` by opening another terminal (on Windows: open Anaconda Prompt instead) and executing:
```bash
cd <PATH_TO_LOCAL_VIVARIUM_REPO>
source env_vivarium/bin/activate
jupyter notebook
```
- This will open a web page in the browser with a list of files. Open the practical session you want to do (`session_1.ipynb` if it is the first class).

The rest of the session is described in this newly opened document, please continue from there. 
Here is a quick overview of the available sessions:

- [Session 1](session_1.ipynb): Introduction to basic of the Notebook controller API
- [Session 2](session_2.ipynb): Defining behaviors definition for agents
- [Session 3](session_3.ipynb): Implementing parallel behaviors and add more sensing abilities
- [Session 4](session_4.ipynb): Modulating internal states and sensing other agent's attributes
- [Session 5](session_5_bonus.ipynb): Understanding routines and creating a simple Eco-Evolutionary simulation
- [Session 6](session_6_logging.ipynb): Logging and plotting data
