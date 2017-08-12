# coding=utf-8

import argparse
import os
import signal
import sys

import settings
from trainers import SegmentationTrainer, SemisupervisedSegmentationTrainer, TrainerBase

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
                                                    debug=debug)
    elif trainer_type == 'classification':
        raise NotImplementedError('Classification training has not yet been implemented')
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))

    signal.signal(signal.SIGINT, get_signal_handler(trainer))
    trainer.train()

if __name__ == "__main__":
    main()
