#!/bin/bash

#Request 1 Tesla K80 gpu(s) and 4 CPU(s) for 2 day(s)
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpu
#SBATCH --mem 32G
#SBATCH -c 4
#SBATCH -t 02-00:00:00
#SBATCH -D /scratch/work/jhnissin/semantic-segmentation/

module purge
module load anaconda2
source activate semantic-segmentation
module load CUDA/8.0.61 cudnn/5.1-CUDA-7.5

srun python -m src.train --model enet-naive-upsampling --mfolder enet-naive-upsampling-supervised --trainer segmentation --config ./configs/config-segmentation-supervised.json --wdir /scratch/work/jhnissin/semantic-segmentation/ --maxjobs 4