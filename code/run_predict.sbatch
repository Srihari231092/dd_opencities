#!/bin/bash

#SBATCH --account=mscagpu
#SBATCH --job-name=satellite
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mscagpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30000


# Load modules:
module load Anaconda3/5.0.1 cuda/9.0
# module load Anaconda3/2019.03

# Activate conda environment:
source activate z11
# source activate tf-gpu-1.14.0

# GPU-based executable:
python predict.py -i "./weights/weights_34567.hdf5" -p 160 -n 5 -o './output/result_34567.tif' -m './output/map_34567.tif' -t '24' -w 0.2 0.3 0.1 0.1 0.3 -bc 3 4 5 6 7
python predict.py -i "./weights/weights_34567.hdf5" -p 160 -n 5 -o './output/result_test_34567.tif' -m './output/map_test_34567.tif' -t 'test' -w 0.2 0.3 0.1 0.1 0.3 -bc 3 4 5 6 7
