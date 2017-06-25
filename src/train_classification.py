# coding=utf-8

import sys
import json
import os
import random
import datetime

from models import model_utils
import dataset_utils

import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, Adam

from PIL import ImageFile
from dataset_utils import ClassificationDataGenerator

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


def get_optimizer():
    optimizer = None

    if get_config_value('optimizer') == 'adam':
        lr = get_config_value('learning_rate')
        decay = get_config_value('decay')
        optimizer = Adam(lr=lr, decay=decay)
        log('Using Adam optimizer with learning rate: {}, decay: {}'.format(lr, decay))
    elif get_config_value('optimizer') == 'sgd':
        lr = get_config_value('learning_rate')
        decay = get_config_value('decay')
        momentum = get_config_value('momentum')
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay)
        log('Using SGD optimizer with learning rate: {}, momentum: {}, decay: {}'.format(lr, momentum, decay))
    else:
        log('Unknown optimizer: {} exiting'.format(get_config_value('optimizer')))
        sys.exit(0)

    return optimizer


def get_callbacks():
    callbacks = []

    # Make sure the model checkpoints directory exists
    create_path_if_not_existing(get_config_value('keras_model_checkpoint_file_path'))

    model_checkpoint_callback = ModelCheckpoint(
        filepath=get_config_value('keras_model_checkpoint_file_path'),
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)

    callbacks.append(model_checkpoint_callback)

    # Tensorboard checkpoint callback to save on every epoch
    if get_config_value('keras_tensorboard_log_path') is not None:
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
    if get_config_value('keras_csv_log_file_path') is not None:
        create_path_if_not_existing(get_config_value('keras_csv_log_file_path'))

        csv_logger_callback = CSVLogger(
            get_config_value('keras_csv_log_file_path'),
            separator=',',
            append=False)

        callbacks.append(csv_logger_callback)

    if get_config_value('reduce_lr_on_plateau') is not None:
        rlr_config = get_config_value('reduce_lr_on_plateau')

        factor = rlr_config.get('factor') or 0.1
        patience = rlr_config.get('patience') or 10
        min_lr = rlr_config.get('min_lr') or 0
        epsilon = rlr_config.get('epsilon') or 0.0001
        cooldown = rlr_config.get('cooldown') or 0
        verbose = rlr_config.get('verbose') or 0

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=factor,
                                      patience=patience,
                                      min_lr=min_lr,
                                      epsilon=epsilon,
                                      cooldown=cooldown,
                                      verbose=verbose)

        callbacks.append(reduce_lr)

    return callbacks


def load_latest_weights(model):
    initial_epoch = 0

    try:
        # Try to find weights from the checkpoint path
        weights_folder = os.path.dirname(get_config_value('keras_model_checkpoint_file_path'))
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

##############################################
# MAIN
##############################################

if __name__ == '__main__':

    # Read the configuration file and make it global
    if len(sys.argv) < 2:
        print 'Invalid number of parameters, usage: python {} <config.json>'.format(sys.argv[0])
        sys.exit(0)

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    CONFIG = read_config_json(sys.argv[1])
    LOG_FILE_PATH = get_config_value('log_file_path')
    print 'Configuration file read successfully'

    # Setup the global LOG_FILE_PATH to enable logging
    log('\n\n############################################################\n')
    log('Starting a new session at local time {}\n'.format(datetime.datetime.now()))

    # Seed the random in order to be able to reproduce the results
    # Note: both random and np.random
    log('Starting program with random seed: {}'.format(get_config_value('random_seed')))
    random.seed(get_config_value('random_seed'))
    np.random.seed(get_config_value('random_seed'))

    # Set image data format
    log('Setting Keras image data format to: {}'.format(get_config_value('image_data_format')))
    K.set_image_data_format(get_config_value('image_data_format'))

    # Create the training data generator
    log('Creating training data generator with labels file: {}'.format(get_config_value('path_to_training_labels_file')))

    training_data_generator = ClassificationDataGenerator(
        path_to_images_folder=get_config_value('path_to_images'),
        per_channel_mean=get_config_value('per_channel_mean'),
        path_to_labels_file=get_config_value('path_to_training_labels_file'),
        path_to_categories_file=get_config_value('path_to_categories_file'),
        verbose=True)

    # Create the validation data generator
    log('Creating validation data generator with labels file: {}'.format(get_config_value('path_to_validation_labels_file')))

    validation_data_generator = ClassificationDataGenerator(
        path_to_images_folder=get_config_value('path_to_images'),
        per_channel_mean=get_config_value('per_channel_mean'),
        path_to_labels_file=get_config_value('path_to_validation_labels_file'),
        path_to_categories_file=get_config_value('path_to_categories_file'),
        verbose=True)

    # Create the model
    model_name = get_config_value('model')
    num_classes = training_data_generator.num_categories
    num_channels = get_config_value('num_channels')
    image_width = get_config_value('image_width')
    image_height = get_config_value('image_height')
    input_shape = (image_height, image_width, num_channels)

    log('Creating model {} instance with input shape: {}, num classes: {}'
        .format(model_name, input_shape, num_classes))

    model = model_utils.get_model(model_name, input_shape, num_classes)
    model.summary()

    log('Compiling model with optimizer: {}, loss function: {}'
        .format(get_config_value('optimizer'), get_config_value('loss_function')))

    # Get the optimizer
    optimizer = get_optimizer()

    model.compile(
        optimizer=optimizer,
        loss=get_config_value('loss_function'),
        metrics=['accuracy'])

    # Load existing weights to continue training
    initial_epoch = 0

    if get_config_value('continue_from_last_checkpoint'):
        initial_epoch = load_latest_weights(model)

    # Get callbacks
    callbacks = get_callbacks()

    num_epochs = get_config_value('num_epochs')
    batch_size = get_config_value('batch_size')
    training_set_size = training_data_generator.num_samples
    validation_set_size = validation_data_generator.num_samples
    steps_per_epoch = training_set_size // batch_size
    validation_steps = validation_set_size // batch_size

    log('Starting training for {} epochs with batch size: {}, steps per epoch: {}, validation steps: {}'
        .format(num_epochs, batch_size, steps_per_epoch, validation_steps))

    model.fit_generator(
        generator=training_data_generator.get_flow(batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        validation_data=validation_data_generator.get_flow(batch_size),
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks)
