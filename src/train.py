# coding=utf-8

import argparse
import os
import numpy as np
import signal
import sys

import settings
from trainers import SegmentationTrainer, SemisupervisedSegmentationTrainer, TrainerBase
from utils import image_utils

early_exit_signal_handler_called = False


def get_signal_handler(trainer):
    # type: (TrainerBase) -> ()

    def signal_handler(signal, frame):
        global early_exit_signal_handler_called

        if not early_exit_signal_handler_called:

            early_exit_signal_handler_called = True

            if trainer is not None:
                trainer.handle_early_exit()
            else:
                print 'No trainer present, exiting'

            sys.exit(0)

    return signal_handler


def ema_smoothing_coefficient_function(step_idx):
    # type: (int) -> float

    """
    Implements a ramp-up period for the Mean Teacher method's EMA smoothing coefficient.

    # Arguments
        :param step_idx: index of the training step
    # Returns
        :return: EMA smoothing coefficient
    """

    # Original paper: 40 000
    if step_idx < 10000:
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
    # Original paper: 40 000
    ramp_up_period = 10000.0

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

    return 5.0


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
    #val = image_utils.np_get_slic_segmentation(normalized_img, 250, sigma=0, compactness=2.0, max_iter=20)
    val = image_utils.np_get_felzenswalb_segmentation(normalized_img, scale=550, sigma=1.5, min_size=20)
    return val


def main():
    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-t', '--trainer', required=True, type=str, choices=['segmentation', 'semisupervised-segmentation', 'classification'], help='Type of the trainer')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-f', '--mfolder', required=True, type=str, help='Name of the model folder')
    ap.add_argument('-w', '--wdir', required=False, type=str, help="Path to working directory")
    ap.add_argument('-d', '--debug', required=False, type=str, help="Path to debug output")
    ap.add_argument('--maxjobs', required=False, type=int, help="Maximum number of parallel jobs (threads/processes)")
    args = vars(ap.parse_args())

    trainer_type = args['trainer']
    trainer_config_file_path = args['config']
    wdir_path = args['wdir']
    model_name = args['model']
    model_folder_name = args['mfolder']
    max_jobs = args['maxjobs']
    debug = args['debug']

    if wdir_path:
        print 'Setting working directory to: {}'.format(wdir_path)
        os.chdir(wdir_path)

    if max_jobs:
        print 'Setting maximum number of parallel jobs to: {}'.format(max_jobs)
        settings.MAX_NUMBER_OF_JOBS = max_jobs

    if debug:
        print 'Running in debug mode, output will be saved to: {}'.format(debug)

    if trainer_type == 'segmentation':
        trainer = SegmentationTrainer(model_name=model_name,
                                      model_folder_name=model_folder_name,
                                      config_file_path=trainer_config_file_path,
                                      debug=debug)
    elif trainer_type == 'semisupervised-segmentation':
        trainer = SemisupervisedSegmentationTrainer(model_name=model_name,
                                                    model_folder_name=model_folder_name,
                                                    config_file_path=trainer_config_file_path,
                                                    debug=debug,
                                                    label_generation_function=label_generation_function,
                                                    consistency_cost_coefficient_function=consistency_coefficient_function,
                                                    ema_smoothing_coefficient_function=ema_smoothing_coefficient_function,
                                                    unlabeled_cost_coefficient_function=unlabeled_cost_coefficient_function)
    elif trainer_type == 'classification':
        raise NotImplementedError('Classification training has not yet been implemented')
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))

    signal.signal(signal.SIGINT, get_signal_handler(trainer))
    trainer.train()

if __name__ == "__main__":
    main()
