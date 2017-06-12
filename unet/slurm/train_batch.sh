#!/bin/bash

#Request 1 Tesla K80 gpus
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpu
#SBATCH --mem-per-cpu 16G
#SBATCH -t 4:00:00

module load anaconda2 CUDA/7.5.18 cudnn/4
source activate /scratch/work/jhnissin/anaconda/environments/semantic-segmentation

python /scratch/work/jhnissin/anaconda/environments/semantic-segmentation/unet/train.py $0