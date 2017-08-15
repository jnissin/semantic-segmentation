#!/bin/bash

sinteractive -t 00:30:00 -p gpushort --gres=gpu:1 --constraint=hsw -c 4 --mem=16G

module purge
module load anaconda2
source activate semantic-segmentation
module load CUDA/8.0.61 cudnn/5.1-CUDA-7.5

cd /scratch/work/jhnissin/semantic-segmentation/