#!/bin/bash

#Request 1 Tesla K80 gpus for 9 hours
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpu
#SBATCH --mem 32G
#SBATCH -t 09:00:00

module load anaconda2 CUDA/7.5.18 cudnn/4
source activate semantic-segmentation

python /scratch/work/jhnissin/anaconda/environments/semantic-segmentation/unet/train.py /scratch/work/jhnissin/anaconda/environments/semantic-segmentation/unet/config/config-filtered.json