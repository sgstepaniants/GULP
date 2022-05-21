#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python compute_dists.py $1 $2