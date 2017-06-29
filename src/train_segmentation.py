# coding=utf-8

import datetime
import random
import sys
import keras

import numpy as np
from PIL import ImageFile
from keras import backend as K

from models import model_utils
from utils import dataset_utils
from utils.dataset_utils import SegmentationDataGenerator
from utils.training_utils import log, get_config_value
import utils.training_utils as training_utils


##############################################
# UTILITIES
##############################################


def get_loss_function(training_set):
    loss_function = None

    if get_config_value('loss_function') == 'pixelwise_crossentropy':
        loss_function = model_utils.pixelwise_crossentropy
        log('Using pixelwise cross-entropy loss function')
    elif get_config_value('loss_function') == 'weighted_pixelwise_crossentropy':

        # Get or calculate the median frequency balancing weights
        median_frequency_balancing_weights = get_config_value('median_frequency_balancing_weights')

        if median_frequency_balancing_weights is None or \
                (len(median_frequency_balancing_weights) != len(material_class_information)):
            log('Median frequency balancing weights were not found or did not match the number of material classes')
            log('Calculating median frequency balancing weights for the training set')

            training_set_masks = [sample[1] for sample in training_set]
            median_frequency_balancing_weights = dataset_utils.calculate_median_frequency_balancing_weights(
                get_config_value('path_to_masks'), training_set_masks, material_class_information)

            log('Median frequency balancing weights calculated: {}'.format(median_frequency_balancing_weights))
        else:
            log('Using existing median frequency balancing weights: {}'.format(median_frequency_balancing_weights))
            median_frequency_balancing_weights = np.array(median_frequency_balancing_weights)

        median_frequency_balancing_weights = K.constant(value=median_frequency_balancing_weights)
        loss_function = model_utils.weighted_pixelwise_crossentropy(median_frequency_balancing_weights)
        log('Using weighted pixelwise cross-entropy loss function with median frequency balancing weights')
    else:
        log('Unknown loss function: {} exiting').format(get_config_value('loss_function'))
        sys.exit(0)

    return loss_function


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

    log('Loading material class information')
    material_class_information = dataset_utils.load_material_class_information(
        get_config_value('path_to_material_class_file'))
    num_classes = len(material_class_information)
    log('Loaded {} material classes successfully'.format(num_classes))

    # Read the data
    log('Reading photo files from: {}'.format(get_config_value('path_to_photos')))
    photo_files = dataset_utils.get_files(get_config_value('path_to_photos'))
    log('Found {} photo files'.format(len(photo_files)))

    log('Reading mask files from: {}'.format(get_config_value('path_to_masks')))
    mask_files = dataset_utils.get_files(get_config_value('path_to_masks'))
    log('Found {} mask files'.format(len(mask_files)))

    if len(photo_files) != len(mask_files):
        raise ValueError(
            'Unmatching photo - mask file list sizes: photos: {}, masks: {}'.format(len(photo_files), len(mask_files)))

    # Generate random splits of the data for training, validation and test
    log('Splitting data to training, validation and test sets of sizes (%) of the whole dataset of size {}: {}'.format(
        len(photo_files), get_config_value('dataset_splits')))
    training_set, validation_set, test_set = dataset_utils.split_dataset(
        photo_files,
        mask_files,
        get_config_value('dataset_splits'))

    log('Dataset split complete')
    log('Training set size: {}'.format(len(training_set)))
    log('Validation set size: {}'.format(len(validation_set)))
    log('Test set size: {}'.format(len(test_set)))

    log('Saving the dataset splits to log file\n')
    log('training_set: {}\n'.format(training_set), False)
    log('validation_set: {}\n'.format(validation_set), False)
    log('test_set: {}\n'.format(test_set), False)

    # Create the model
    model_name = get_config_value('model')
    num_channels = get_config_value('num_channels')
    input_shape = (None, None, num_channels)

    log('Creating model {} instance with input shape: {}, num classes: {}'
        .format(model_name, input_shape, num_classes))

    model = model_utils.get_model(model_name, input_shape, num_classes)
    model.summary()

    # Look for a crop size
    crop_size = None

    if get_config_value('crop_width') is None or get_config_value('crop_height') is None:
        crop_size = None
    else:
        crop_size = (get_config_value('crop_width'), get_config_value('crop_height'))

    use_data_augmentation = bool(get_config_value('use_data_augmentation'))

    log('Creating training data generator')
    training_data_generator = SegmentationDataGenerator(
        photo_files_folder_path=get_config_value('path_to_photos'),
        mask_files_folder_path=get_config_value('path_to_masks'),
        photo_mask_files=training_set,
        material_class_information=material_class_information,
        num_channels=num_channels,
        random_seed=get_config_value('random_seed'),
        per_channel_mean_normalization=True,
        per_channel_mean=get_config_value('per_channel_mean'),
        per_channel_stddev_normalization=True,
        per_channel_stddev=get_config_value('per_channel_stddev'),
        use_data_augmentation=use_data_augmentation,
        augmentation_probability=0.5,
        rotation_range=40.0,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=False)

    log('Creating validation data generator')
    validation_data_generator = SegmentationDataGenerator(
        photo_files_folder_path=get_config_value('path_to_photos'),
        mask_files_folder_path=get_config_value('path_to_masks'),
        photo_mask_files=validation_set,
        material_class_information=material_class_information,
        num_channels=num_channels,
        random_seed=get_config_value('random_seed'),
        per_channel_mean_normalization=True,
        per_channel_mean=training_data_generator.per_channel_mean,
        per_channel_stddev_normalization=True,
        per_channel_stddev=training_data_generator.per_channel_stddev,
        use_data_augmentation=False)

    log('Using per-channel mean: {}'.format(training_data_generator.per_channel_mean))
    log('Using per-channel stddev: {}'.format(training_data_generator.per_channel_stddev))

    # Get callbacks
    callbacks = training_utils.get_callbacks(
        keras_model_checkpoint_file_path=get_config_value('keras_model_checkpoint_file_path'),
        keras_tensorboard_log_path=get_config_value('keras_tensorboard_log_path'),
        keras_csv_log_file_path=get_config_value('keras_csv_log_file_path'),
        reduce_lr_on_plateau=get_config_value('reduce_lr_on_plateau'))

    initial_epoch = 0

    # Load existing weights to continue training
    if get_config_value('continue_from_last_checkpoint'):
        initial_epoch = training_utils.load_latest_weights(
            get_config_value('keras_model_checkpoint_file_path'),
            model)

    # Transfer weights
    if get_config_value('transfer_weights'):
        if initial_epoch != 0:
            log('Cannot transfer weights when continuing from last checkpoint. Skipping weight transfer')
        else:
            transfer_weights_options = get_config_value('transfer_weights_options')
            transfer_model_name = transfer_weights_options['transfer_model_name']
            transfer_model_input_shape = tuple(transfer_weights_options['transfer_model_input_shape'])
            transfer_model_num_classes = transfer_weights_options['transfer_model_num_classes']
            transfer_model_weights_file_path = transfer_weights_options['transfer_model_weights_file_path']

            log('Creating transfer model: {} with input shape: {}, num classes: {}'
                .format(transfer_model_name, transfer_model_input_shape, transfer_model_num_classes))
            transfer_model = model_utils.get_model(transfer_model_name, transfer_model_input_shape, transfer_model_num_classes)
            transfer_model.summary()

            log('Loading transfer weights to transfer model from file: {}'.format(transfer_model_weights_file_path))
            transfer_model.load_weights(transfer_model_weights_file_path)

            from_layer_index = transfer_weights_options['from_layer_index']
            to_layer_index = transfer_weights_options['to_layer_index']
            freeze_transferred_layers = transfer_weights_options['freeze_transferred_layers']
            log('Transferring weights from layer range: [{}:{}], freeze transferred layers: {}'
                .format(from_layer_index, to_layer_index, freeze_transferred_layers))

            transferred_layers, last_transferred_layer = training_utils.transfer_weights(
                from_model=transfer_model,
                to_model=model,
                from_layer_index=from_layer_index,
                to_layer_index=to_layer_index,
                freeze_transferred_layers=freeze_transferred_layers)

            log('Weight transfer completed with {} transferred layers, last transferred layer: {}'
                .format(transferred_layers, last_transferred_layer))

    # Get the optimizer
    optimizer = training_utils.get_optimizer(get_config_value('optimizer'))

    # Get the loss function
    loss_function = get_loss_function(training_set=training_set)

    # Compile the model - note: must be compiled after weight transfer in order for
    # possible layer freezing to take effect
    log('Compiling the model')
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', model_utils.mean_iou(num_classes), model_utils.mean_per_class_accuracy(num_classes)])

    num_epochs = get_config_value('num_epochs')
    batch_size = get_config_value('batch_size')
    training_set_size = len(training_set)
    validation_set_size = len(validation_set)
    steps_per_epoch = training_set_size // batch_size
    validation_steps = validation_set_size // batch_size

    log('Starting training for {} epochs with batch size: {}, crop_size: {}, steps per epoch: {}, validation steps: {}'
        .format(num_epochs, batch_size, crop_size, steps_per_epoch, validation_steps))

    model.fit_generator(
        generator=training_data_generator.get_flow(batch_size, crop_size),
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        validation_data=validation_data_generator.get_flow(batch_size, crop_size),
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks)

    log('The session ended at local time {}\n'.format(datetime.datetime.now()))

    # Close the log file
    if training_utils.LOG_FILE:
        training_utils.LOG_FILE.close()
