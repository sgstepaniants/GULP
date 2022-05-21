#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python fit_model.py $1 $2