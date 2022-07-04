#!/bin/sh
##############################################################################
#   .,**ooooo***.       ',***********      ',************          *****.
#  .ooooooooooooo       '*ooooooooooo      .ooooooooooooo         'oooooo.
#  *ooo.''''',ooo       '*o*.''''''''       '''',ooo.''''        '*oo,.*oo.
# '*ooo       '''       '*oo*,,,,,,,,           .ooo'           '*oo,  .oo*'
# '*ooo                 '*oooooooooo*           .ooo'           ,oo,    ooo*'
# '*ooo      ',**'      '*o*.''''''''           .ooo'          .ooo*****oooo*
#  ,ooo*,,,,,*ooo'      '*oo*,,,,,,,,           .ooo'         .oooooooooooooo
#  '*oooooooooooo       '*ooooooooooo'          .ooo'         oooo,''''''.*oo.
#    '.,,,,,,,,.'        .,.........,           '.,,          ,.,.        '...
#
#                          CETA-CIEMAT GPU CLUSTER
#                   ***************************************
#
#                        Example 02: MPI + CUDA example
#
##############################################################################

#SBATCH --nodes=2
#SBATCH --partition=gpu.prod
#SBATCH --gres=gpu:2
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

conda init --all
conda activate kepler
module load cuda


python3 REINFORCE_map.py ./graphs/$1

exit 0
