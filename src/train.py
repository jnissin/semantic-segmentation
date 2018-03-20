# coding=utf-8

import argparse
import signal
import sys
import multiprocessing
import os
import resource

import settings

from utils.multiprocessing_utils import MultiprocessingManager

_EARLY_EXIT_SIGNAL_HANDLER_CALLED = multiprocessing.Value('i', 0)
_EARLY_EXIT_SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGABRT, signal.SIGQUIT]
_MAIN_PROCESS_PID = multiprocessing.Value('i', -1)
_TRAINER = None


def signal_handler(s, f):

    global _MAIN_PROCESS_PID, _EARLY_EXIT_SIGNAL_HANDLER_CALLED, _TRAINER
    process_pid = multiprocessing.current_process().pid
    print 'Received signal: {} in process {} - main process pid: {}'.format(s, process_pid, _MAIN_PROCESS_PID.value)

    if _EARLY_EXIT_SIGNAL_HANDLER_CALLED.value == 0 and (process_pid == _MAIN_PROCESS_PID.value or _MAIN_PROCESS_PID.value == -1):
        _EARLY_EXIT_SIGNAL_HANDLER_CALLED.value = 1

        if _TRAINER is not None:
            _TRAINER.handle_early_exit()
        else:
            print 'No trainer present, exiting'

        sys.exit(0)
    else:
        print 'Not the main process - waiting for parent process to join'


def main():
    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-t', '--trainer', required=True, type=str,
                    choices=['segmentation_supervised',
                             'segmentation_supervised_mean_teacher',
                             'segmentation_semi_supervised_mean_teacher',
                             'segmentation_semi_supervised_superpixel',
                             'segmentation_semi_supervised_mean_teacher_superpixel',
                             'classification_supervised',
                             'classification_supervised_mean_teacher',
                             'classification_semi_supervised_mean_teacher'], help='Type of the trainer')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-f', '--mfolder', required=True, type=str, help='Name of the model folder')
    ap.add_argument('-w', '--wdir', required=False, type=str, help="Path to working directory")
    ap.add_argument('--maxjobs', required=False, type=int, help="Maximum number of parallel jobs (threads/processes)")
    ap.add_argument('--maxmemory', required=False, type=int, help="Maximum memory available to this and child processes (GB)")
    args = vars(ap.parse_args())

    trainer_type = args['trainer']
    trainer_super_type = trainer_type.split('_')[0]
    trainer_config_file_path = args['config']
    wdir_path = args['wdir']
    model_name = args['model']
    model_folder_name = args['mfolder']
    max_jobs = args['maxjobs']
    max_memory = args['maxmemory']

    if settings.DEBUG:
        print 'RUNNING IN DEBUG MODE'

    if settings.PROFILE:
        print 'RUNNING IN PROFILE MODE'

    global _MAIN_PROCESS_PID, _TRAINER

    if _MAIN_PROCESS_PID.value == -1:
        pid = multiprocessing.current_process().pid
        print 'Storing main process pid: {}'.format(pid)
        _MAIN_PROCESS_PID.value = pid

    if wdir_path:
        print 'Setting working directory to: {}'.format(wdir_path)
        os.chdir(wdir_path)

    if max_jobs:
        print 'Setting maximum number of parallel jobs to: {}'.format(max_jobs)
        settings.MAX_NUMBER_OF_JOBS = max_jobs

    if max_memory:
        print 'Setting maximum memory limit (soft) to: {} (GB)'.format(max_memory)
        rsrc = resource.RLIMIT_DATA
        soft, hard = resource.getrlimit(rsrc)
        print 'Current memory limits: soft: {}, hard: {}'.format(soft, hard)

        # Only set the soft limit - breaking the hard limit crashes the software
        max_memory_soft_kb = max_memory * 1048576
        resource.setrlimit(rsrc, (max_memory_soft_kb, -1))
        soft, hard = resource.getrlimit(rsrc)
        print 'New memory limits: soft: {}, hard: {}'.format(soft, hard)

    if settings.USE_MULTIPROCESSING:
        print 'Instantiating MultiprocessingManager'
        mp = MultiprocessingManager()
        print 'MultiprocessingManager instantiated - currently hosting: {} clients'.format(mp.num_current_clients)

    # Register early exit signal handlers
    for sig in _EARLY_EXIT_SIGNALS:
        print 'Registering early exit signal handler for signal: {}'.format(sig)
        signal.signal(sig, signal_handler)

    if trainer_super_type == 'segmentation':
        from trainers import SegmentationTrainer

        _TRAINER = SegmentationTrainer(trainer_type=trainer_type,
                                       model_name=model_name,
                                       model_folder_name=model_folder_name,
                                       config_file_path=trainer_config_file_path)
    elif trainer_super_type == 'classification':
        from trainers import ClassificationTrainer

        _TRAINER = ClassificationTrainer(trainer_type=trainer_type,
                                         model_name=model_name,
                                         model_folder_name=model_folder_name,
                                         config_file_path=trainer_config_file_path)
    else:
        raise ValueError('Unsupported trainer type: {}'.format(trainer_type))

    history = _TRAINER.train()
    sys.exit(0)


if __name__ == "__main__":
    main()
