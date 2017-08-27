# coding=utf-8

import argparse
import os
import signal
import sys

import settings
from trainers import SegmentationTrainer, SegmentationTrainer, TrainerBase

_EARLY_EXIT_SIGNAL_HANDLER_CALLED = False


def get_signal_handler(trainer):
    # type: (TrainerBase) -> ()

    def signal_handler(signal, frame):
        global _EARLY_EXIT_SIGNAL_HANDLER_CALLED
        if not _EARLY_EXIT_SIGNAL_HANDLER_CALLED:

            _EARLY_EXIT_SIGNAL_HANDLER_CALLED = True

            if trainer is not None:
                trainer.handle_early_exit()
            else:
                print 'No trainer present, exiting'

            sys.exit(0)

    return signal_handler


def main():
    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-t', '--trainer', required=True, type=str,
                    choices=['segmentation_supervised',
                             'segmentation_semi_supervised_mean_teacher',
                             'segmentation_semi_supervised_superpixel',
                             'segmentation_semi_supervised_mean_teacher_superpixel',
                             'classification_supervised',
                             'classification_semi_supervised_mean_teacher'], help='Type of the trainer')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-f', '--mfolder', required=True, type=str, help='Name of the model folder')
    ap.add_argument('-w', '--wdir', required=False, type=str, help="Path to working directory")
    ap.add_argument('--maxjobs', required=False, type=int, help="Maximum number of parallel jobs (threads/processes)")
    args = vars(ap.parse_args())

    trainer_type = args['trainer']
    trainer_super_type = trainer_type.split('_')[0]
    trainer_config_file_path = args['config']
    wdir_path = args['wdir']
    model_name = args['model']
    model_folder_name = args['mfolder']
    max_jobs = args['maxjobs']

    if wdir_path:
        print 'Setting working directory to: {}'.format(wdir_path)
        os.chdir(wdir_path)

    if max_jobs:
        print 'Setting maximum number of parallel jobs to: {}'.format(max_jobs)
        settings.MAX_NUMBER_OF_JOBS = max_jobs

    if settings.DEBUG:
        print 'Running in debug mode'

    if settings.PROFILE:
        print 'Running in profile mode'

    if trainer_super_type == 'segmentation':
        trainer = SegmentationTrainer(trainer_type=trainer_type,
                                      model_name=model_name,
                                      model_folder_name=model_folder_name,
                                      config_file_path=trainer_config_file_path)
    elif trainer_super_type == 'classification':
        raise ValueError('Classification trainer is disabled for the time of refactoring')
#        trainer = ClassificationTrainer(trainer_type=trainer_type,
#                                        model_name=model_name,
#                                        model_folder_name=model_folder_name,
#                                        config_file_path=trainer_config_file_path)
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))

    signal.signal(signal.SIGINT, get_signal_handler(trainer))
    signal.signal(signal.SIGTERM, get_signal_handler(trainer))
    history = trainer.train()


if __name__ == "__main__":
    main()
