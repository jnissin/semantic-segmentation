#!/bin/bash

sinteractive -t 00:30:00 -p gpushort --gres=gpu:1 --constraint=hsw -c 8 --mem=42G

module purge
module load anaconda2
module load teflon
source activate ss-new
module load CUDA/9.0.176 cuDNN/7-CUDA-9.0.176

cd /scratch/work/jhnissin/semantic-segmentation/