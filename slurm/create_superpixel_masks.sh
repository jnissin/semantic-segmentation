#!/bin/bash

#SBATCH --mem 18G
#SBATCH -c 20
#SBATCH -t 01-00:00:00
#SBATCH -D /scratch/work/jhnissin/semantic-segmentation/

module purge
module load anaconda2
module load teflon
source activate ss-new

cd /scratch/work/jhnissin/semantic-segmentation/

python -m src.scripts.create_superpixel_masks --photos data/final/unlabeled/ --output data/final/unlabeled_fzw_masks/ --function felzenswalb --jobs 20 --verbose true --dtype float --equalize true --iexisting true --bonly true --bconnectivity 2