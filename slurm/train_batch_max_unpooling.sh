#!/bin/bash

#Request 1 Tesla K80 gpus and 2 CPUs for 9 hours
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpushort
#SBATCH --mem 32G
#SBATCH -c 2
#SBATCH -t 04:00:00

module purge
module load anaconda2
source activate semantic-segmentation
module load CUDA/8.0.61 cudnn/5

cd /scratch/work/jhnissin/semantic-segmentation/src
srun python /scratch/work/jhnissin/semantic-segmentation/src/train_segmentation.py /scratch/work/jhnissin/semantic-segmentation/configs/config-filtered-max-unpooling.json