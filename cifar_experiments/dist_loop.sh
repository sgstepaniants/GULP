#!/bin/bash 

# Run the script
declare -i batches=250

for (( i = 1; i <= $batches; i++ ))
do
    sbatch -n 4 -o distcifar-$i-$batches.log distcifar_script.sh $i $batches
done