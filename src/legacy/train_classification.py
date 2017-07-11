# coding=utf-8

import datetime
import random
import sys

import keras
import keras.backend as K

import numpy as np
from PIL import ImageFile

import utils.training_utils as training_utils
from models import model_utils

from utils.dataset_utils import ClassificationDataGenerator
from utils.training_utils import log, get_config_value

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

    training_utils.CONFIG = training_utils.read_config_json(sys.argv[1])
    training_utils.LOG_FILE_PATH = get_config_value('log_file_path')
    print 'Configuration file read successfully'

    # Setup the global LOG_FILE_PATH to enable logging
    log('\n\n############################################################\n')
    log('Starting a new session at local time {}\n'.format(datetime.datetime.now()))
    log('Using keras version: {}'.format(keras.__version__))
    log('Using tensorflow version: {}'.format(K.tf.__version__))

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
        num_channels=get_config_value('num_channels'),
        verbose=True)

    # Create the validation data generator
    log('Creating validation data generator with labels file: {}'.format(get_config_value('path_to_validation_labels_file')))

    validation_data_generator = ClassificationDataGenerator(
        path_to_images_folder=get_config_value('path_to_images'),
        per_channel_mean=get_config_value('per_channel_mean'),
        path_to_labels_file=get_config_value('path_to_validation_labels_file'),
        path_to_categories_file=get_config_value('path_to_categories_file'),
        num_channels=get_config_value('num_channels'),
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
        .format(get_config_value('optimizer')['name'], get_config_value('loss_function')))

    # Get the optimizer
    optimizer = training_utils.get_optimizer(get_config_value('optimizer'))

    model.compile(
        optimizer=optimizer,
        loss=get_config_value('loss_function'),
        metrics=['accuracy'])

    # Load existing weights to continue training
    initial_epoch = 0

    if get_config_value('continue_from_last_checkpoint'):
        initial_epoch = training_utils.load_latest_weights(
            get_config_value('keras_model_checkpoint_file_path'),
            model)

    # Get callbacks
    callbacks = training_utils.get_callbacks(
        keras_model_checkpoint_file_path=get_config_value('keras_model_checkpoint_file_path'),
        keras_tensorboard_log_path=get_config_value('keras_tensorboard_log_path'),
        keras_csv_log_file_path=get_config_value('keras_csv_log_file_path'),
        reduce_lr_on_plateau=get_config_value('reduce_lr_on_plateau'))

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

    log('The session ended at local time {}\n'.format(datetime.datetime.now()))

    # Close the log file
    if training_utils.LOG_FILE:
        training_utils.LOG_FILE.close()