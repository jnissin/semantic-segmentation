# coding=utf-8

import argparse
import os
import numpy as np

from trainers import SegmentationTrainer, SemisupervisedSegmentationTrainer
from utils import image_utils


def ema_smoothing_coefficient_function(step_idx):
    # type: (int) -> float

    """
    Implements a ramp-up period for the Mean Teacher method's EMA smoothing coefficient.

    # Arguments
        :param step_idx: index of the training step
    # Returns
        :return: EMA smoothing coefficient
    """

    if step_idx < 40000:
        a = 0.999
    else:
        a = 0.99

    return a


def consistency_coefficient_function(step_idx):
    # type: (int) -> float

    """
    Implements a ramp-up period for the Mean Teacher method's consistency coefficient.

    # Arguments
        :param step_idx: index of the training step
    # Returns
        :return: consistency coefficient for the given step
    """

    # type: (int) -> float
    # How many steps for the ramp up period
    ramp_up_period = 40000.0

    # x exists [0,1]
    x = float(step_idx)/ramp_up_period
    x = min(x, 1.0)
    return np.exp(-5.0*((1.0-x)**2))


def unlabeled_cost_coefficient_function(step_idx):
    # type: (int) -> float

    """
    Returns the unlabeled data cost coefficient for the given training step.

    # Arguments
        :param step_idx: index of the training step
    # Return
        :return: unlabeled cost coefficient
    """

    return 0.9


def label_generation_function(np_img):
    # type: (np.array[float]) -> (np.array[int])

    """
    Generates superpixel segmentation (labels) for the given image. The image is expected to be
    in an unnormalized form with color values in range [0, 255]. The superpixel labels are in
    index encoded format (integers) with array shape HxW, where the entries are superpixel
    indices. The integer range is continuous in range [0, num_superpixels]. All the images do
    not necessarily have the same number of superpixels, but the number can be calculated by
    taking the max from the data.

    # Arguments
        :param np_img: image as a numpy array
    # Returns
        :return: superpixel segmentation
    """
    normalized_img = image_utils.np_normalize_image_channels(np_img, clamp_to_range=True)
    val = image_utils.np_get_superpixel_segmentation(normalized_img, 200)
    return val


def main():
    # Construct the argument parser and parge arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-t', '--trainer', required=True, type=str, choices=['segmentation', 'semisupervised-segmentation', 'classification'], help='Type of the trainer')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-f', '--mfolder', required=True, type=str, help='Name of the model folder')
    ap.add_argument('-w', '--wdir', required=False, type=str, help="Path to working directory")
    args = vars(ap.parse_args())

    trainer_type = args['trainer']
    trainer_config_file_path = args['config']
    wdir_path = args['wdir']
    model_name = args['model']
    model_folder_name = args['mfolder']

    if wdir_path:
        print 'Setting working directory to: {}'.format(wdir_path)
        os.chdir(wdir_path)

    if trainer_type == 'segmentation':
        trainer = SegmentationTrainer(model_name=model_name,
                                      model_folder_name=model_folder_name,
                                      config_file_path=trainer_config_file_path)
        trainer.train()
    elif trainer_type == 'semisupervised-segmentation':
        trainer = SemisupervisedSegmentationTrainer(model_name=model_name,
                                                    model_folder_name=model_folder_name,
                                                    config_file_path=trainer_config_file_path,
                                                    label_generation_function=label_generation_function,
                                                    consistency_cost_coefficient_function=consistency_coefficient_function,
                                                    ema_smoothing_coefficient_function=ema_smoothing_coefficient_function,
                                                    unlabeled_cost_coefficient_function=unlabeled_cost_coefficient_function)
        trainer.train()
    elif trainer_type == 'classification':
        raise NotImplementedError('Classification training has not yet been implemented')
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))


if __name__ == "__main__":
    main()