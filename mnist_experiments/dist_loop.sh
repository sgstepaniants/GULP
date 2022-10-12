#!/bin/bash 

# Run the script
declare -i batches=275

for (( i = 1; i <= $batches; i++ ))
do
    sbatch -n 4 -o distcomp-$i-$batches.log dist_script.sh $i $batches
done
