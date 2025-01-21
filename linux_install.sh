#!/bin/bash

# clone the project repository and navigate to it  
git clone https://github.com/flowersteam/vivarium.git
cd vivarium/

# install venv module if not already done
sudo apt install -y python3-venv

# create a virtual environment
python3 -m venv env_vivarium

# activate the virtual environment
source env_vivarium/bin/activate

# install the required packages
pip install -r requirements.txt
pip install -e . 
