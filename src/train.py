# coding=utf-8

import argparse
import numpy as np

from trainers import SegmentationTrainer, SemisupervisedSegmentationTrainer
from generators import DataAugmentationParameters


def get_data_augmentation_parameters():
    data_augmentation_params = DataAugmentationParameters(
        augmentation_probability=0.5,
        rotation_range=40.0,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=False)

    return data_augmentation_params


def ema_smoothing_coefficient_function(step_idx):
    # type: (int) -> float

    if step_idx < 40000:
        a = 0.999
    else:
        a = 0.99

    return a


def consistency_coefficient_function(step_idx):
    # type: (int) -> float
    # How many steps for the ramp up period
    ramp_up_period = 40000.0

    # x exists [0,1]
    x = float(step_idx)/ramp_up_period
    x = min(x, 1.0)
    return np.exp(-5.0*((1.0-x)**2))


def main():
    # Construct the argument parser and parge arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-t', '--trainer', required=True, type=str,
                    choices=['segmentation', 'semisupervised-segmentation', 'classification'],
                    help='Type of the trainer')
    ap.add_argument('-c', '--config', required=True, type=str,
                    help='Path to trainer configuration JSON file')
    args = vars(ap.parse_args())

    trainer_type = args['trainer']
    trainer_config_file_path = args['config']
    data_augmentation_params = get_data_augmentation_parameters()

    if trainer_type == 'segmentation':
        trainer = SegmentationTrainer(config_file_path=trainer_config_file_path,
                                      data_augmentation_parameters=data_augmentation_params)
        trainer.train()
    elif trainer_type == 'semisupervised-segmentation':
        trainer = SemisupervisedSegmentationTrainer(config_file_path=trainer_config_file_path,
                                                    data_augmentation_parameters=data_augmentation_params,
                                                    consistency_cost_coefficient_function=consistency_coefficient_function,
                                                    ema_smoothing_coefficient_function=ema_smoothing_coefficient_function,
                                                    lambda_loss_function=None)
        trainer.train()
    elif trainer_type == 'classification':
        raise NotImplementedError('Classification training has not yet been implemented')
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))


if __name__ == "__main__":
    main()
