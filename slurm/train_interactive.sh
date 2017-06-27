#!/bin/bash

module purge
module load anaconda2
source activate semantic-segmentation
module load CUDA/8.0.61 cudnn/5

sinteractive -t 00:30:00 -p gpushort --gres=gpu:teslak80:1 --mem=32G
