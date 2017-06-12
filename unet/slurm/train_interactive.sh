#!/bin/bash

module load anaconda2 CUDA/7.5.18 cudnn/4
source activate semantic-segmentation

sinteractive -t 0:30:00 -p gpushort --gres=gpu:teslak80:1