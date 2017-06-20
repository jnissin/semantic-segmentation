#!/bin/bash

#Request 1 Tesla K80 gpus and 2 CPUs for 9 hours
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpushort
#SBATCH --mem 32G
#SBATCH -c 2
#SBATCH -t 04:00:00

module load anaconda2 CUDA/7.5.18 cudnn/4
source activate semantic-segmentation

cd /scratch/work/jhnissin/semantic-segmentation/src
srun python train.py /scratch/work/jhnissin/semantic-segmentation/src/configs/config-filtered.json