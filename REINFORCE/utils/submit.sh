#!/bin/sh

conda init --all
conda activate kepler
module load cuda


# Launch benchmarks:


benchmarks=("graph_6.json")

# Time to finish
t="01:00:00"


for b in "${benchmarks[@]}"; do

  echo -e " -J M"$b" -p kepler -t $t --nodes=1 --exclusive --gres=gpu:2 --wait-all-nodes=1 ./launch.sh "$b

  sbatch    -J M"$b" -p kepler -t $t --nodes=1 --exclusive --gres=gpu:2 --wait-all-nodes=1 ./launch.sh $b

done
