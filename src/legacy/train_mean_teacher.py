"""
Implements mean teacher semi-supervised training
"""

import keras
import time
import random
import sys
import datetime

import numpy as np

from PIL import ImageFile
from keras import backend as K
from keras.preprocessing.image import list_pictures

from utils.training_utils import log, get_config_value
from utils import dataset_utils
from utils import training_utils
from models import model_utils


def get_consistency_cost_coefficient(x):
    # type: (float) -> float

    """
    Calculates the consistency cost coefficient described in the
    original Mean Teacher paper:

        https://arxiv.org/pdf/1703.01780.pdf

    # Arguments
        :param x: progress of the training [0,1]
    # Returns
        :return: consistency cost coefficient
    """

    return np.exp(-5.0*((1.0-x)**2))


def get_random_batch_data(batch_size,
                          x_shape,
                          y_shape):
    # type: (np.array) -> (np.array, np.array)

    """
    Generates random data for testing purposes. Assumes that the label data (Y) is one hot
    encoded in the final axis.

    # Arguments
        :param batch_size: size of the batch
        :param x_shape: shape of the random input data
        :param y_shape: shape of the random label data
    # Returns
        :return: random batch data as two numpy arrays (X, Y)
    """

    # Get random input data between [-1, 1]
    X = np.random.random(size=list([batch_size])+list(x_shape))
    X -= 0.5
    X *= 2.0

    # Get random one hot encoded array
    Y = np.zeros(shape=list([batch_size])+list(y_shape))

    for i in range(0, batch_size):
        for j in range(0, y_shape[0]):
            for k in range(0, y_shape[1]):
                entry = np.zeros(shape=y_shape[-1])
                entry[np.random.randint(0, y_shape[-1])] = 1
                Y[i, j, k] = entry

    return X, Y


def get_random_batch_data2(batch_size, x_shape, y_shape, student_model, teacher_model):
    # type: (numpy.array) -> numpy.array

    """
    Generates random data for testing purposes. Assumes that the label data (Y) is one hot
    encoded in the final axis.

    # Arguments
        :param batch_size: size of the batch
        :param x_shape: shape of the random input data
        :param y_shape: shape of the random label data
    # Returns
        :return: random batch data as two numpy arrays (X, Y)
    """

    training_step_index = 0
    while True:
        if not training_step_index == 0:
            # Perform EMA update
            perform_ema_teacher_update(teacher_model=teacher_model,
                                       student_model=student_model,
                                       training_step=training_step_index,
                                       verbose=True)

        # Get random input data between [-1, 1]
        X = np.random.random(size=list([batch_size])+list(x_shape))
        X -= 0.5
        X *= 2.0

        # Get random one hot encoded array
        Y = np.zeros(shape=list([batch_size])+list(y_shape))

        for i in range(0, batch_size):
            for j in range(0, y_shape[0]):
                for k in range(0, y_shape[1]):
                    entry = np.zeros(shape=y_shape[-1])
                    entry[np.random.randint(0, y_shape[-1])] = 1
                    Y[i, j, k] = entry


        # Predict on the batch using the teacher
        teacher_predictions = teacher_model.predict_on_batch(X)

        # Get the consistency cost coefficient and expand it to batch sized numpy array
        consistency_cost_coeff = np.ones([batch_size]) * get_consistency_cost_coefficient(0.5)

        # Create a new input array from the original input batch, teacher predictions and consistency cost coefficient
        inputs = {
            "images": X,
            "labels": Y,
            "mt_predictions": teacher_predictions,
            "consistency_cost": consistency_cost_coeff
        }

        dummy_output = np.zeros([batch_size])

        outputs = {
            "loss": dummy_output
        }

        training_step_index += 1
        yield inputs, outputs


def perform_ema_teacher_update(teacher_model, student_model, training_step, verbose=False):
    # type: (keras.models.Model, keras.models.Model, int) -> ()

    """
    Performs the Exponential Moving Average weight update on the teacher model. This should be
    called after the student has updated its weights on the batch. Uses a similar setup for
    ramp-up as the SVHN in the original Mean Teacher paper:

        https://arxiv.org/pdf/1703.01780.pdf

    # Arguments
        :param teacher_model: the teacher model
        :param student_model: the student model
        :param training_step: the number of the training step starting from one (determines ramp-up)
    # Returns
        :return: nothing
    """
    s_time = time.time()

    if training_step < 40000:
        a = 0.999
    else:
        a = 0.99

    # Perform the weight update: theta'_t = a * theta'_t-1 + (1 - a) * theta_t
    t_weights = teacher_model.get_weights()
    s_weights = student_model.get_weights()

    if len(t_weights) != len(s_weights):
        raise ValueError('The weight arrays are not of the same length for the student and teacher: {} vs {}'
                         .format(len(t_weights), len(s_weights)))

    num_weights = len(t_weights)

    for i in range(0, num_weights):
        t_weights[i] = a * t_weights[i] + (1.0 - a) * s_weights[i]

    if verbose:
        print 'EMA update finished in {} seconds for {} layers\' weights'.format(time.time()-s_time, num_weights)


def initialize_program(configuration_file_path):

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    training_utils.CONFIG = training_utils.read_config_json(configuration_file_path)
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


def load_material_set():
    # type: () -> (tuple(list(MaterialClassInformation, int)))

    """
    Loads the material information from a file and returns the list of MaterialClassInformation and
    number of classes.

    # Arguments
        None
    # Returns
        :return: list of material classes and number of material classes
    """

    log('Loading material class information')
    material_class_information = dataset_utils.load_material_class_information(
        get_config_value('path_to_material_class_file'))
    num_classes = len(material_class_information)
    log('Loaded {} material classes successfully'.format(num_classes))

    return material_class_information, num_classes


def load_data_set():
    """
    Locates all the data in the semi-supervised dataset and returns the paths to every file
    in their respective lists.

    # Arguments
        None
    # Returns
        :return: three lists of file paths: labeled photos, labeled masks, unlabeled photos
    """

    log('Reading labeled photo files from: {}'.format(get_config_value('path_to_labeled_photos')))
    labeled_photo_files = list_pictures(get_config_value('path_to_labeled_photos'))
    log('Found {} labeled photo files'.format(len(labeled_photo_files)))

    log('Reading labeled mask files from: {}'.format(get_config_value('path_to_labeled_masks')))
    labeled_mask_files = list_pictures(get_config_value('path_to_labeled_masks'))
    log('Found {} labeled mask files'.format(len(labeled_mask_files)))

    if len(labeled_photo_files) != len(labeled_mask_files):
        raise ValueError('Unmatching labeled photo - labeled mask file list sizes: photos: {}, masks: {}'
                         .format(len(labeled_photo_files), len(labeled_mask_files)))

    log('Reading unlabeled photo files from: {}'.format(get_config_value('path_to_unlabeled_photos')))
    unlabeled_photo_files = list_pictures(get_config_value('path_to_unlabeled_photos'))
    log('Found {} unlabeled photo files'.format(len(unlabeled_photo_files)))

    return labeled_photo_files, labeled_mask_files, unlabeled_photo_files


def split_dataset(labeled_photo_files, labeled_mask_files):
    """
    Splits the labeled data into training, validation and test sets. Logs the splits for
    reproducibility.

    # Arguments
        :param labeled_photo_files: paths to the labeled photo files
        :param labeled_mask_files: paths to the labeled mask files
    # Return
        :return: three lists of (photo, mask) file pairs: training, validation and test
    """

    # Generate random splits of the supervised data for training, validation and test
    log('Splitting data to training, validation and test sets of sizes (%) of the labeled dataset of size {}: {}'
        .format(len(labeled_photo_files), get_config_value('dataset_splits')))

    labeled_training_set, labeled_validation_set, labeled_test_set = \
        dataset_utils.split_labeled_dataset(labeled_photo_files, labeled_mask_files, get_config_value('dataset_splits'))

    log('Dataset split complete')
    log('Training set size: {}'.format(len(labeled_training_set)))
    log('Validation set size: {}'.format(len(labeled_validation_set)))
    log('Test set size: {}'.format(len(labeled_test_set)))

    log('Saving the dataset splits to log file\n')
    log('training_set: {}\n'.format(labeled_training_set), False)
    log('validation_set: {}\n'.format(labeled_validation_set), False)
    log('test_set: {}\n'.format(labeled_test_set), False)

    return labeled_training_set, labeled_validation_set, labeled_test_set


if __name__ == '__main__':

    """
    INPUT_SHAPE = (256, 256, 3)
    NUM_CLASSES = 24
    LABEL_SHAPE = (256, 256, NUM_CLASSES)
    BATCH_SIZE = 2

    # Instantiate and compile the student model
    student_model = unet.get_segnet_mean_teacher(INPUT_SHAPE, NUM_CLASSES, trainable=True)
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    student_model.compile(loss={'loss': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    student_model.summary()

    # Instantiate the teacher model
    teacher_model = unet.get_segnet_mean_teacher(INPUT_SHAPE, NUM_CLASSES, trainable=False)

    # Copy student weights to teacher
    teacher_model.set_weights(student_model.get_weights())

    # Start training
    print 'Starting training'
    training_step_index = 0
    dummy_output = np.zeros(shape=[BATCH_SIZE])

    while True:
        print 'Starting training step: {}'.format(training_step_index+1)

        X, Y = get_random_batch_data(BATCH_SIZE, INPUT_SHAPE, LABEL_SHAPE)

        # Predict on the batch using the teacher
        teacher_predictions = teacher_model.predict_on_batch(X)

        # Get the consistency cost coefficient and expand it to batch sized numpy array
        consistency_cost_coeff = np.ones(shape=[BATCH_SIZE]) * get_consistency_cost_coefficient(0.5)

        # Create a new input array from the original input batch, teacher predictions and consistency cost coefficient
        inputs = {
            "images": X,
            "labels": Y,
            "mt_predictions": teacher_predictions,
            "consistency_cost": consistency_cost_coeff
        }

        outputs = {
            "loss": dummy_output
        }

        # Train the student on batch
        training_loss = student_model.train_on_batch(x=inputs, y=outputs)

        # Perform EMA update
        perform_ema_teacher_update(teacher_model=teacher_model,
                                   student_model=student_model,
                                   training_step=training_step_index+1,
                                   verbose=True)

        # Update the training step index
        training_step_index += 1
    """

    # Program initialization
    if len(sys.argv) < 2:
        print 'Invalid number of parameters, usage: python {} <config.json>'.format(sys.argv[0])
        sys.exit(0)

    initialize_program(sys.argv[1])

    # Data set processing
    material_class_information, num_classes = load_material_set()
    labeled_photo_files, labeled_mask_files, unlabeled_photo_files = load_data_set()
    labeled_training_set, labeled_validation_set, labeled_test_set = split_dataset(labeled_photo_files, labeled_mask_files)

    # Model creation
    use_mean_teacher_method = bool(get_config_value('use_mean_teacher'))
    log('Use mean teacher method for training: {}'.format(use_mean_teacher_method))

    model_name = get_config_value('model')
    num_channels = get_config_value('num_channels')
    input_shape = get_config_value('input_shape')

    # TODO: Create student/teacher model support to get_model
    log('Creating student model {} instance with input shape: {}, num classes: {}'.format(model_name, input_shape, num_classes))
    student_model = model_utils.get_model(model_name, input_shape, num_classes, student_model=use_mean_teacher_method)
    student_model.summary()

    # TODO: Create student/teacher model support to get_model
    teacher_model = None

    if use_mean_teacher_method:
        log('Creating teacher model {} instance with input shape: {}, num classes: {}'.format(model_name, input_shape, num_classes))
        teacher_model = model_utils.get_model(model_name, input_shape, num_classes, student_model=False)
        teacher_model.summary()

    # Read data generation parameters
    crop_shape = get_config_value('crop_shape')
    use_data_augmentation = bool(get_config_value('use_data_augmentation'))

    data_augmentation_params = DataAugmentationParameters(
        augmentation_probability=0.5,
        rotation_range=40.0,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=False)

    # Create training data and validation data generators
    # Note: training data comes from semi-supervised segmentation data generator and validation
    # and test data come from regular segmentation data generator
    log('Creating training data generator')
    training_data_generator = SemisupervisedSegmentationDataGenerator(
        labeled_data=labeled_training_set,
        unlabeled_photo_files=unlabeled_photo_files,
        material_class_information=material_class_information,
        num_channels=num_channels,
        random_seed=get_config_value('random_seed'),
        per_channel_mean_normalization=True,
        per_channel_mean=get_config_value('per_channel_mean'),
        per_channel_stddev_normalization=True,
        per_channel_stddev=get_config_value('per_channel_stddev'),
        use_data_augmentation=use_data_augmentation,
        data_augmentation_params=data_augmentation_params)

    log('Creating validation data generator')
    validation_data_generator = SegmentationDataGenerator(
        labeled_data=labeled_validation_set,
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
    callbacks = training_utils._get_training_callbacks(
        keras_model_checkpoint_file_path=get_config_value('keras_model_checkpoint_file_path'),
        keras_tensorboard_log_path=get_config_value('keras_tensorboard_log_path'),
        keras_csv_log_file_path=get_config_value('keras_csv_log_file_path'),
        reduce_lr_on_plateau=get_config_value('reduce_lr_on_plateau'),
        optimizer_checkpoint_file_path=get_config_value('optimizer_checkpoint_file_path'))

    initial_epoch = 0

    # Load existing weights to continue training
    if get_config_value('continue_from_last_checkpoint'):
        initial_epoch = training_utils.load_latest_weights(
            get_config_value('keras_model_checkpoint_file_path'),
            student_model)

    # Get the optimizer
    # Note: we will only provide the optimizer configuration file from previous run
    # if we are continuing training
    optimizer = training_utils._get_model_optimizer(get_config_value('optimizer'),
                                                    get_config_value('optimizer_checkpoint_file_path') if initial_epoch != 0 else None)

    # Get the loss function
    loss_function = get_loss_function(training_set=training_set)

    # Compile the model - note: must be compiled after weight transfer in order for
    # possible layer freezing to take effect
    # TODO: How to account for the unlabeled data in the mIoU and mpca calculation? pass the number of unlabeled
    # similar to weighted pixelwise cross-entropy calculation?
    log('Compiling the student model')
    student_model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', model_utils.mean_iou(num_classes), model_utils.mean_per_class_accuracy(num_classes)])

    num_epochs = get_config_value('num_epochs')
    batch_size = get_config_value('batch_size')
    training_set_size = len(training_set)
    validation_set_size = len(validation_set)
    training_steps = dataset_utils.get_number_of_batches(training_set_size, batch_size)
    validation_steps = dataset_utils.get_number_of_batches(validation_set_size, batch_size)

    log('Starting training for {} epochs with batch size: {}, crop_size: {}, training steps epoch: {}, validation steps: {}'
        .format(num_epochs, batch_size, crop_size, training_steps, validation_steps))

    model.fit_generator(
        generator=training_data_generator.get_flow(batch_size, crop_size),
        steps_per_epoch=training_steps,
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