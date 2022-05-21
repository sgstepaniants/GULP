#!/bin/bash 

# Run the script
declare -i batches=250

for (( i = 1; i <= $batches; i++ ))
do
    sbatch -n 4 -o results-$i-$batches.log fit_script.sh $i $batches
done
