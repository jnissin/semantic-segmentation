# coding=utf-8

import os
import json
import numpy as np

import dataset_utils as dataset_utils
from ..callbacks.optimizer_checkpoint import OptimizerCheckpoint

import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, Optimizer
from keras.models import Model

##############################################
# GLOBALS
##############################################


CONFIG = None
LOG_FILE = None
LOG_FILE_PATH = None


##############################################
# UTILITIES
##############################################


def log(s, log_to_stdout=True):
    global LOG_FILE
    global LOG_FILE_PATH

    # Create and open the log file
    if not LOG_FILE:
        if LOG_FILE_PATH:
            create_path_if_not_existing(LOG_FILE_PATH)

            LOG_FILE = open(LOG_FILE_PATH, 'w')
        else:
            raise ValueError('The log file path is None, cannot log')

    # Log to file - make sure there is a newline
    if not s.endswith('\n'):
        LOG_FILE.write(s + "\n")
    else:
        LOG_FILE.write(s)

    # Log to stdout - no newline needed
    if log_to_stdout:
        print s.strip()


def create_path_if_not_existing(path):
    if not path:
        return

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def read_config_json(path):
    with open(path) as f:
        data = f.read()
        return json.loads(data)


def get_config_value(key):
    global CONFIG
    return CONFIG[key] if key in CONFIG else None


def set_config_value(key, value):
    global CONFIG
    CONFIG[key] = value


def get_latest_weights_file_path(weights_folder_path):
    weight_files = dataset_utils.get_files(weights_folder_path)

    if len(weight_files) > 0:
        weight_files.sort()
        weight_file = weight_files[-1]

        if os.path.isfile(os.path.join(weights_folder_path, weight_file)) and weight_file.endswith(".hdf5"):
            return os.path.join(weights_folder_path, weight_file)

    return None


def load_latest_weights(weights_directory_path, model):
    initial_epoch = 0

    try:
        # Try to find weights from the checkpoint path
        weights_folder = os.path.dirname(weights_directory_path)
        log('Searching for existing weights from checkpoint path: {}'.format(weights_folder))
        weight_file_path = get_latest_weights_file_path(weights_folder)

        if weight_file_path is None:
            log('Could not locate any suitable weight files from the given path')
            return 0

        weight_file = weight_file_path.split('/')[-1]

        if weight_file:
            log('Loading network weights from file: {}'.format(weight_file_path))
            model.load_weights(weight_file_path)

            # Parse the epoch number: <epoch>-<vloss>
            epoch_val_loss = weight_file.split('.')[1]
            initial_epoch = int(epoch_val_loss.split('-')[0]) + 1
            log('Continuing training from epoch: {}'.format(initial_epoch))
        else:
            log('No existing weights were found')

    except Exception as e:
        log('Searching for existing weights finished with an error: {}'.format(e.message))
        return 0

    return initial_epoch
