# coding=utf-8

import os
import json
import numpy as np

import dataset_utils as dataset_utils

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
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


def get_optimizer(optimizer_info):
    optimizer = None

    optimizer_name = optimizer_info['name']

    if optimizer_name == 'adam':
        lr = optimizer_info['learning_rate']
        decay = optimizer_info['decay']
        optimizer = Adam(lr=lr, decay=decay)
        log('Using Adam optimizer with learning rate: {}, decay: {}'.format(lr, decay))
    elif optimizer_name == 'sgd':
        lr = optimizer_info['learning_rate']
        decay = optimizer_info['decay']
        momentum = optimizer_info['momentum']
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay)
        log('Using SGD optimizer with learning rate: {}, momentum: {}, decay: {}'.format(lr, momentum, decay))
    else:
        raise ValueError('Unknown optimizer name: {}'.format(optimizer_name))

    return optimizer


def get_callbacks(keras_model_checkpoint_file_path,
                  keras_tensorboard_log_path=None,
                  keras_csv_log_file_path=None,
                  reduce_lr_on_plateau=None):
    callbacks = []

    # Make sure the model checkpoints directory exists
    create_path_if_not_existing(keras_model_checkpoint_file_path)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=keras_model_checkpoint_file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)

    callbacks.append(model_checkpoint_callback)

    # Tensorboard checkpoint callback to save on every epoch
    if keras_tensorboard_log_path is not None:
        create_path_if_not_existing(get_config_value('keras_tensorboard_log_path'))

        tensorboard_checkpoint_callback = TensorBoard(
            log_dir=get_config_value('keras_tensorboard_log_path'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None)

        callbacks.append(tensorboard_checkpoint_callback)

    # CSV logger for streaming epoch results
    if keras_csv_log_file_path is not None:
        create_path_if_not_existing(get_config_value('keras_csv_log_file_path'))

        csv_logger_callback = CSVLogger(
            get_config_value('keras_csv_log_file_path'),
            separator=',',
            append=False)

        callbacks.append(csv_logger_callback)

    if reduce_lr_on_plateau is not None:
        factor = reduce_lr_on_plateau.get('factor') or 0.1
        patience = reduce_lr_on_plateau.get('patience') or 10
        min_lr = reduce_lr_on_plateau.get('min_lr') or 0
        epsilon = reduce_lr_on_plateau.get('epsilon') or 0.0001
        cooldown = reduce_lr_on_plateau.get('cooldown') or 0
        verbose = reduce_lr_on_plateau.get('verbose') or 0

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=factor,
                                      patience=patience,
                                      min_lr=min_lr,
                                      epsilon=epsilon,
                                      cooldown=cooldown,
                                      verbose=verbose)

        callbacks.append(reduce_lr)

    return callbacks


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


def transfer_weights(from_model, to_model, from_layer_index, to_layer_index, freeze_transferred_layers):
    # type: (Model, Model, int, int, bool) -> (int, str)
    num_transferred_layers = 0

    # Support negative indexing
    if from_layer_index < 0:
        from_layer_index += len(from_model.layers)

    if to_layer_index < 0:
        to_layer_index += len(from_model.layers)

    # Assumes indexing is the same for both models for the specified
    # layer range
    for i in range(from_layer_index, to_layer_index):
        to_model.layers[i].set_weights(from_model.layers[i].get_weights())

        if freeze_transferred_layers:
            to_model.layers[i].trainable = False

        num_transferred_layers += 1

    return num_transferred_layers, from_model.layers[to_layer_index-1].name
