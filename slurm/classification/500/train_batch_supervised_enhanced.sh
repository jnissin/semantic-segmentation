#!/bin/bash

# Request 1 Tesla K80 or P100. The 'hsw' constraint guarantees K80 or P100 because they are the only GPU nodes with hsw processors
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --constraint=hsw
#SBATCH --mem 24G
#SBATCH -c 6
#SBATCH -t 01-00:00:00
#SBATCH -D /scratch/work/jhnissin/semantic-segmentation/

module purge
module load anaconda2
module load teflon
source activate ss-new
module load CUDA/9.0.176 cuDNN/7-CUDA-9.0.176

srun python -m src.train --model enet-naive-upsampling-enhanced-encoder-only --mfolder enet-naive-upsampling-enhanced-encoder-only/500/supervised --trainer classification_supervised --config ./configs/classification/config-classification-500.json --wdir /scratch/work/jhnissin/semantic-segmentation/ --maxjobs 6