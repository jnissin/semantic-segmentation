#!/bin/bash

# Request 1 Tesla K80 or P100. The 'hsw' constraint guarantees K80 or P100 because they are the only GPU nodes with hsw processors
#SBATCH --gres=gpu:1
#SBATCH --constraint=hsw
#SBATCH -p gpu
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -t 02-00:00:00
#SBATCH -D /scratch/work/jhnissin/semantic-segmentation/

module purge
module load anaconda2
source activate semantic-segmentation
module load CUDA/8.0.61 cudnn/5.1-CUDA-7.5

srun python -m src.train --model enet-naive-upsampling --mfolder enet-naive-upsampling/supervised --trainer segmentation_supervised --config ./configs/config-segmentation.json --wdir /scratch/work/jhnissin/semantic-segmentation/ --maxjobs 4