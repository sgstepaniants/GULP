#!/bin/bash 

# Run the script
declare -i batches=180

for (( i = 1; i <= $batches; i++ ))
do
    sbatch -n 4 -o rep-$i-$batches.log rep_script.sh $i $batches
done
