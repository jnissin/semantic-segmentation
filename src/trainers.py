# coding = utf-8

import os
import json
import random
import datetime
import time
import shutil
import numpy as np

from enum import Enum
from PIL import ImageFile
from abc import ABCMeta, abstractmethod, abstractproperty

import keras
import keras.backend as K
from keras.optimizers import Optimizer
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from keras_extensions.extended_optimizers import SGD, Adam
from keras_extensions.extended_model import ExtendedModel

from tensorflow.python.client import timeline

from utils import dataset_utils
from utils import image_utils
from utils import general_utils

from callbacks.optimizer_checkpoint import OptimizerCheckpoint
from callbacks.stepwise_learning_rate_scheduler import StepwiseLearningRateScheduler
from generators import DataGenerator, SegmentationDataGenerator, MINCDataSet, ClassificationDataGenerator
from generators import DataGeneratorParameters, SegmentationDataGeneratorParameters, DataAugmentationParameters
from enums import BatchDataFormat, SuperpixelSegmentationFunctionType

from logger import Logger
from data_set import LabeledImageDataSet, UnlabeledImageDataSet
from losses import ModelLambdaLossType

import models
import losses
import metrics
import settings


#############################################
# TIMELINER (PROFILING)
#############################################


class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


#############################################
# TRAINER
#############################################

class TrainerType(Enum):
    SEGMENTATION_SUPERVISED = 0
    SEGMENTATION_SUPERVISED_MEAN_TEACHER = 1
    SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER = 2
    SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL = 3
    SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL = 4
    CLASSIFICATION_SUPERVISED = 5
    CLASSIFICATION_SUPERVISED_MEAN_TEACHER = 6
    CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER = 7


class TrainerBase:
    """
    An abstract base class that implements methods shared between different
    types of trainers, e.g. SupervisedTrainer, SemisupervisedSegmentationTrainer
    or ClassificationTrainer.
    """

    __metaclass__ = ABCMeta

    def __init__(self, trainer_type, model_name, model_folder_name, config_file_path):
        # type: (str, str, str, str) -> None

        """
        Initializes the trainer i.e. seeds random, loads material class information and
        data sets etc.

        # Arguments
            :param trainer_type: type of the trainer
            :param model_name: name of the NN model to instantiate
            :param model_folder_name: name of the model folder (for saving data)
            :param config_file_path: path to the configuration file
        # Returns
            Nothing
        """
        # Declare instance variables
        self.trainer_type = TrainerType[trainer_type.upper()]
        self.model_name = model_name
        self.model_folder_name = model_folder_name
        self._log_folder_path = None
        self.config = None
        self.logger = None

        self.training_set_labeled = None
        self.training_set_unlabeled = None
        self.validation_set = None
        self.training_data_generator = None
        self.validation_data_generator = None
        self.model_wrapper = None

        self._initial_epoch = None
        self._initial_step = None
        self.last_completed_epoch = -1

        # Profiling related variables
        self.profiling_timeliner = TimeLiner() if settings.PROFILE else None
        self.profiling_run_metadata = K.tf.RunMetadata() if settings.PROFILE else None
        self.profiling_run_options = K.tf.RunOptions(trace_level=K.tf.RunOptions.FULL_TRACE) if settings.PROFILE else None

        # Without this some truncated images can throw errors
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.config = self._load_config_json(config_file_path)
        self.save_values_on_early_exit = self._get_config_value('save_values_on_early_exit')

        # Setup the log file path to enable logging
        log_file_path = self._populate_path_template(self._get_config_value('log_file_path'))
        log_to_stdout = self._get_config_value('log_to_stdout')
        self.logger = Logger(log_file_path=log_file_path, use_timestamp=True, log_to_stdout_default=log_to_stdout)

        # Log the Keras and Tensorflow versions
        self.logger.log('############################################################\n\n')
        self.logger.log('Using Keras version: {}'.format(keras.__version__))
        self.logger.log('Using Tensorflow version: {}'.format(K.tf.__version__))

        # Create a copy of the config file for future reference
        copy_config_file_path = os.path.join(self.logger.log_folder_path, os.path.basename(config_file_path))
        self.logger.log('Creating a copy of the used configuration file to model data directory: {}'.format(copy_config_file_path))
        shutil.copy(config_file_path, copy_config_file_path)

        # Seed the random in order to be able to reproduce the results
        # Note: both random and np.random
        self.logger.log('Initializing random and np.random with random seed: {}'.format(self.random_seed))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Set image data format
        self.logger.log('Setting Keras image data format to: {}'.format(self.image_data_format))
        K.set_image_data_format(self.image_data_format)

        # Get data augmentation parameters
        self.data_augmentation_parameters = self._get_data_augmentation_parameters()

        # Initialize data sets
        self.training_set_labeled, self.training_set_unlabeled, self.validation_set = self._get_data_sets()

        # Initialize data generators
        self.training_data_generator, self.validation_data_generator = self._get_data_generators()

        # Get the iterator references
        self.training_data_iterator = self.training_data_generator.get_data_set_iterator()
        self.validation_data_iterator = self.validation_data_generator.get_data_set_iterator()

        # Initialize model
        self.model_wrapper = self._init_model()

    def _init_model(self):
        # type: () -> ModelBase

        """
        Initializes the models i.e. creates the model and initiali

        # Arguments
            None

        # Returns
            :return: The ModelBase instance of the model
        """

        self.logger.log('Initializing model')

        # Model creation
        model_lambda_loss_type = self._get_model_lambda_loss_type()

        self.logger.log('Creating model {} instance with type: {}, input shape: {}, num classes: {}'
                        .format(self.model_name, model_lambda_loss_type, self.input_shape, self.num_classes))

        self.model_wrapper = models.get_model(self.model_name,
                                              self.input_shape,
                                              self.num_classes,
                                              logger=self.logger,
                                              model_lambda_loss_type=model_lambda_loss_type)

        self.model.summary()

        if self.continue_from_last_checkpoint:
            self._load_latest_weights_for_model(self.model, self.model_checkpoint_directory)

            if self.training_data_generator is None or self.training_data_iterator.num_steps_per_epoch < 1:
                raise ValueError('Cannot determine initial step - training data generator is not initialized')

        if self.use_transfer_weights:
            if self.initial_epoch != 0:
                self.logger.warn('Should not transfer weights when continuing from last checkpoint. Skipping weight transfer')
                lr_scalers = dict()
            else:
                lr_scalers = self._transfer_weights(to_model_wrapper=self.model_wrapper)
        else:
            lr_scalers = dict()

        # Get the necessary components to compile the model
        model_optimizer = self._get_model_optimizer(lr_scalers)
        model_loss = self._get_model_loss()
        model_loss_weights = self._get_model_loss_weights()
        model_metrics = self._get_model_metrics()

        # Compile the model
        self.model.compile(optimizer=model_optimizer,
                           loss=model_loss,
                           loss_weights=model_loss_weights,
                           metrics=model_metrics,
                           **self._get_compile_kwargs())

        # Log the model structure to a file using the keras plot_model
        try:
            model_plot_file_path = os.path.join(self.log_folder_path, 'model.png')
            self.logger.log('Saving model plot to file: {}'.format(model_plot_file_path))
            plot_model(self.model, to_file=model_plot_file_path, show_shapes=True, show_layer_names=True)
        except Exception as e:
            self.logger.warn('Saving model plot to file failed: {}'.format(e.message))

        return self.model_wrapper

    @abstractmethod
    def _get_data_sets(self):
        # type: () -> (object, object, object)

        """
        Creates and initializes the data sets and return a tuple of three data sets:
        (training labeled, training unlabele and validation)

        # Arguments
            None
        # Returns
            :return: A tuple of data sets (training labeled, training unlabeled, validation)
        """
        pass

    @abstractmethod
    def _get_data_generators(self):
        # type: () -> (DataGenerator, DataGenerator)

        """
        Creates and initializes the data generators and returns a tuple of two data generators:
        (training data generator, validation data generator)

        # Arguments
            None
        # Returns
            :return: A tuple of two data generators (training, validation)
        """
        pass

    """
    PROPERTIES
    """

    @abstractproperty
    def num_classes(self):
        # type: () -> int
        pass

    @abstractproperty
    def per_channel_mean(self):
        # type: () -> np.ndarray
        pass

    @abstractproperty
    def per_channel_stddev(self):
        # type: () -> np.ndarray
        pass

    @property
    def image_data_format(self):
        # type: () -> str
        return self._get_config_value('image_data_format')

    @property
    def random_seed(self):
        # type: () -> int
        return int(self._get_config_value('random_seed'))

    @property
    def continue_from_last_checkpoint(self):
        # type: () -> bool
        return bool(self._get_config_value('continue_from_last_checkpoint'))

    @property
    def initial_epoch(self):
        # type: () -> int
        if self._initial_epoch is None:
            self._initial_epoch = 0

            if self.continue_from_last_checkpoint:
                try:
                    weight_file_path = self._get_latest_weights_file_path(self.model_checkpoint_directory)

                    if weight_file_path is not None:
                        weight_file = weight_file_path.split('/')[-1]

                        if weight_file:
                            # Parse the epoch number: <str>.<epoch>-<val_loss>.<str>
                            epoch_val_loss = weight_file.split('.')[1].rsplit('-', 1)
                            # Initial epoch is the next epoch so add one
                            self._initial_epoch = int(epoch_val_loss[0]) + 1
                            print 'Initial epoch: {}, file: {}'.format(self._initial_epoch, weight_file_path)
                except Exception as e:
                    self._initial_epoch = 0

        return self._initial_epoch

    @property
    def initial_step(self):
        # type: () -> int
        if self._initial_step is None:
            if self.training_data_iterator is None:
                raise ValueError('Training data iterator is not set')

            self._initial_step = self._initial_epoch * self.training_data_iterator.num_steps_per_epoch
        return self._initial_step

    @property
    def continue_from_optimizer_checkpoint(self):
        # type: () -> bool
        return bool(self._get_config_value('continue_from_last_checkpoint'))

    @property
    def optimizer_checkpoint_file_path(self):
        # type: () -> str
        return self._populate_path_template(self._get_config_value('optimizer_checkpoint_file_path'))

    @property
    def log_folder_path(self):
        # type: () -> str
        if self._log_folder_path is None:
            # Log folder can have the model folder within its path
            log_folder_path = self._get_config_value('log_folder_path').format(model_folder=self.model_folder_name)
            log_folder_path = os.path.dirname(log_folder_path) if os.path.isfile(log_folder_path) else log_folder_path

            # If the log folder path already exists create a different path to avoid overwriting logs
            if os.path.exists(log_folder_path):
                log_folder_path = '{}-{:%Y-%m-%d-%H:%M}'.format(os.path.dirname(log_folder_path), datetime.datetime.now())
                print 'Log directory path already exists. Avoiding overwriting by attempting to switch to a new log path: {}'.format(log_folder_path)

            self._log_folder_path = log_folder_path

        return self._log_folder_path

    @property
    def input_shape(self):
        # type: () -> list
        return self._get_config_value('input_shape')

    @property
    def use_class_weights(self):
        # type: () -> bool
        return bool(self._get_config_value('use_class_weights'))

    @property
    def use_transfer_weights(self):
        # type: () -> bool
        return bool(self._get_config_value('transfer_weights'))

    @property
    def transfer_options(self):
        # type: () -> dict
        return self._get_config_value('transfer_options')

    @property
    def loss_function_name(self):
        # type: () -> str
        return self._get_config_value('loss_function')

    @property
    def use_data_augmentation(self):
        # type: () -> bool
        return bool(self._get_config_value('use_data_augmentation'))

    @property
    def num_color_channels(self):
        # type: () -> int
        return int(self._get_config_value('num_color_channels'))

    @property
    def div2_constraint(self):
        # type: () -> int
        val = self._get_config_value('div2_constraint')
        return int(val) if val is not None else 0

    @property
    def num_epochs(self):
        # type: () -> int
        return int(self._get_config_value('num_epochs')) if not settings.PROFILE else settings.PROFILE_NUM_EPOCHS

    @property
    def num_labeled_per_batch(self):
        # type: () -> int
        return int(self._get_config_value('num_labeled_per_batch'))

    @property
    def num_unlabeled_per_batch(self):
        # type: () -> int
        return int(self._get_config_value('num_unlabeled_per_batch'))

    @property
    def crop_shape(self):
        # type: () -> list
        return self._get_config_value('crop_shape')

    @property
    def resize_shape(self):
        # type: () -> list
        return self._get_config_value('resize_shape')

    @property
    def validation_num_labeled_per_batch(self):
        # type: () -> int
        return int(self._get_config_value('validation_num_labeled_per_batch'))

    @property
    def validation_crop_shape(self):
        # type: () -> list
        return self._get_config_value('validation_crop_shape')

    @property
    def validation_resize_shape(self):
        # type: () -> list
        return self._get_config_value('validation_resize_shape')

    @property
    def model(self):
        # type: () -> ExtendedModel
        if self.model_wrapper is not None:
            return self.model_wrapper.model

        return None

    @property
    def training_steps_per_epoch(self):
        # type: () -> int
        if self.training_data_iterator is None:
            raise ValueError('Training data iterator has not been initialized')
        return self.training_data_iterator.num_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH

    @property
    def validation_steps_per_epoch(self):
        # type: () -> int
        if self.validation_data_iterator is None:
            raise ValueError('Validation data iterator has not been initialized')
        return self.validation_data_iterator.num_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH

    def _load_config_json(self, path):
        with open(path) as f:
            data = f.read()
            return json.loads(data)

    def _get_config_value(self, key):
        return self.config[key] if key in self.config else None

    def _set_config_value(self, key, value):
        self.config[key] = value

    def _populate_path_template(self, path, *args, **kwargs):
        # type: (str) -> str
        return path.format(model_folder=self.model_folder_name, log_folder_path=self.log_folder_path, **kwargs)

    def _get_data_augmentation_parameters(self):
        if self._get_config_value('use_data_augmentation'):
            self.logger.log('Parsing data augmentation parameters')

            augmentation_config = self._get_config_value('data_augmentation_params')

            if not augmentation_config:
                raise ValueError('No data with key data_augmentation_params was found in the configuration file')

            augmentation_probability_function = augmentation_config.get('augmentation_probability_function')
            rotation_range = augmentation_config.get('rotation_range')
            zoom_range = augmentation_config.get('zoom_range')
            width_shift_range = augmentation_config.get('width_shift_range')
            height_shift_range = augmentation_config.get('height_shift_range')
            channel_shift_range = augmentation_config.get('channel_shift_range')
            horizontal_flip = augmentation_config.get('horizontal_flip')
            vertical_flip = augmentation_config.get('vertical_flip')
            gaussian_noise_stddev_function = augmentation_config.get('gaussian_noise_stddev_function')
            gamma_adjust_range = augmentation_config.get('gamma_adjust_range')
            mean_teacher_noise_params = augmentation_config.get('mean_teacher_noise_params')

            data_augmentation_parameters = DataAugmentationParameters(
                augmentation_probability_function=augmentation_probability_function,
                rotation_range=rotation_range,
                zoom_range=zoom_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                channel_shift_range=channel_shift_range,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                gaussian_noise_stddev_function=gaussian_noise_stddev_function,
                gamma_adjust_range=gamma_adjust_range,
                mean_teacher_noise_params=mean_teacher_noise_params)

            self.logger.log('Data augmentation params: augmentation probability function: {}, rotation range: {}, zoom range: {}, '
                            'width shift range: {}, height shift range: {}, channel shift range: {}, horizontal flip: {}, vertical flip: {}, '
                            'gaussian noise stddev function: {}, gamma_adjust_range: {}'
                             .format(augmentation_probability_function,
                                     rotation_range,
                                     zoom_range,
                                     width_shift_range,
                                     height_shift_range,
                                     channel_shift_range,
                                     horizontal_flip,
                                     vertical_flip,
                                     gaussian_noise_stddev_function,
                                     gamma_adjust_range))

            self.logger.log('Data augmentation params: mean teacher noise params: {}'.format(mean_teacher_noise_params))
        else:
            data_augmentation_parameters = None

        return data_augmentation_parameters

    def _get_training_callbacks(self):
        keras_model_checkpoint = self._get_config_value('keras_model_checkpoint')
        keras_tensorboard_log_path = self._populate_path_template(self._get_config_value('keras_tensorboard_log_path'))
        keras_csv_log_file_path = self._populate_path_template(self._get_config_value('keras_csv_log_file_path'))
        early_stopping = self._get_config_value('early_stopping')
        reduce_lr_on_plateau = self._get_config_value('reduce_lr_on_plateau')
        stepwise_learning_rate_scheduler = self._get_config_value('stepwise_learning_rate_scheduler')
        optimizer_checkpoint_file_path = self._populate_path_template(self._get_config_value('optimizer_checkpoint_file_path'))

        callbacks = []

        # Always ensure that the model checkpoint has been provided
        if not keras_model_checkpoint:
            raise ValueError('Could not find Keras ModelCheckpoint configuration with key keras_model_checkpoint - would be unable to save model weights')

        # Make sure the model checkpoints directory exists
        keras_model_checkpoint_dir = self._populate_path_template(os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')))
        keras_model_checkpoint_file = os.path.basename(keras_model_checkpoint.get('checkpoint_file_path'))
        keras_model_checkpoint_file_path = os.path.join(keras_model_checkpoint_dir, keras_model_checkpoint_file)
        general_utils.create_path_if_not_existing(keras_model_checkpoint_file_path)
        keras_model_checkpoint_monitor = keras_model_checkpoint.get('monitor') or 'val_loss'
        keras_model_checkpoint_verbose = keras_model_checkpoint.get('verbose') or 1
        keras_model_checkpoint_save_best_only = keras_model_checkpoint.get('save_best_only') or False
        keras_model_checkpoint_save_weights_only = keras_model_checkpoint.get('save_weights_only') or False
        keras_model_checkpoint_mode = keras_model_checkpoint.get('mode') or 'auto'
        keras_model_checkpoint_period = keras_model_checkpoint.get('period') or 1

        model_checkpoint_callback = ModelCheckpoint(
            filepath=keras_model_checkpoint_file_path,
            monitor=keras_model_checkpoint_monitor,
            verbose=keras_model_checkpoint_verbose,
            save_best_only=keras_model_checkpoint_save_best_only,
            save_weights_only=keras_model_checkpoint_save_weights_only,
            mode=keras_model_checkpoint_mode,
            period=keras_model_checkpoint_period)

        callbacks.append(model_checkpoint_callback)

        # Tensorboard checkpoint callback to save on every epoch
        if keras_tensorboard_log_path is not None:
            # Tensorboard log files have unique names for each run - so no worries about overwriting
            general_utils.create_path_if_not_existing(keras_tensorboard_log_path)

            tensorboard_checkpoint_callback = TensorBoard(
                log_dir=keras_tensorboard_log_path,
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                write_grads=False,  # Note: writing grads for a bit network takes about an hour
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None)

            callbacks.append(tensorboard_checkpoint_callback)

        # CSV logger for streaming epoch results
        if keras_csv_log_file_path is not None:

            # Don't overwrite existing CSV files
            if os.path.exists(keras_csv_log_file_path):
                new_keras_csv_log_file_path = '{}-{:%Y-%m-%d-%H:%M}'.format(keras_csv_log_file_path, datetime.datetime.now())
                self.logger.warn('Previous keras_csv_log_file_path log already exists in: {} - switching to path: {}'.format(keras_csv_log_file_path, new_keras_csv_log_file_path))
                keras_csv_log_file_path = new_keras_csv_log_file_path

            general_utils.create_path_if_not_existing(keras_csv_log_file_path)

            csv_logger_callback = CSVLogger(
                keras_csv_log_file_path,
                separator=',',
                append=False)

            callbacks.append(csv_logger_callback)

        # Early stopping to conserve resources
        if early_stopping is not None:
            monitor = early_stopping.get('monitor') or 'val_loss'
            min_delta = early_stopping.get('min_delta') or 0.0
            patience = early_stopping.get('patience') or 2
            verbose = early_stopping.get('verbose') or 0
            mode = early_stopping.get('mode') or 'auto'

            early_stop = EarlyStopping(
                monitor=monitor,
                min_delta=min_delta,
                patience=patience,
                verbose=verbose,
                mode=mode)

            callbacks.append(early_stop)

        # Reduce LR on plateau to adjust learning rate
        if reduce_lr_on_plateau is not None:
            factor = reduce_lr_on_plateau.get('factor') or 0.1
            patience = reduce_lr_on_plateau.get('patience') or 10
            min_lr = reduce_lr_on_plateau.get('min_lr') or 0
            epsilon = reduce_lr_on_plateau.get('epsilon') or 0.0001
            cooldown = reduce_lr_on_plateau.get('cooldown') or 0
            verbose = reduce_lr_on_plateau.get('verbose') or 0
            monitor = reduce_lr_on_plateau.get('monitor') or 'val_loss'

            reduce_lr = ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                epsilon=epsilon,
                cooldown=cooldown,
                verbose=verbose)

            callbacks.append(reduce_lr)

        # Stepwise learning rate scheduler to schedule learning rate ramp-ups/downs
        # Note: This is added after the possible Reduce LR on plateau callback
        # so it overrides it's changes if they are both present
        if stepwise_learning_rate_scheduler is not None:
            lr_schedule = stepwise_learning_rate_scheduler.get('lr_schedule')
            b2_schedule = stepwise_learning_rate_scheduler.get('b2_schedule')
            last_scheduled_step = stepwise_learning_rate_scheduler.get('last_scheduled_step')
            verbose = stepwise_learning_rate_scheduler.get('verbose')

            stepwise_lr_scheduler = StepwiseLearningRateScheduler(lr_schedule=eval(lr_schedule) if lr_schedule is not None else None,
                                                                  b2_schedule=eval(b2_schedule) if b2_schedule is not None else None,
                                                                  last_scheduled_step=last_scheduled_step,
                                                                  initial_step=self.initial_step,
                                                                  verbose=verbose)

            callbacks.append(stepwise_lr_scheduler)

        # Optimizer checkpoint
        if optimizer_checkpoint_file_path is not None:
            general_utils.create_path_if_not_existing(optimizer_checkpoint_file_path)
            optimizer_checkpoint = OptimizerCheckpoint(optimizer_checkpoint_file_path)
            callbacks.append(optimizer_checkpoint)

        return callbacks

    def _get_model_optimizer(self, lr_scalers={}):
        # type: (dict) -> Optimizer

        optimizer_info = self._get_config_value('optimizer')
        optimizer_configuration = None
        optimizer = None
        optimizer_name = optimizer_info['name'].strip().lower()

        if self.continue_from_optimizer_checkpoint and self.initial_epoch == 0:
            self.logger.warn('Cannot continue from optimizer checkpoint if initial epoch is 0. Ignoring optimizer checkpoint.')
        elif self.continue_from_optimizer_checkpoint and self.initial_epoch != 0:
            optimizer_configuration_file_path = self.optimizer_checkpoint_file_path
            self.logger.log('Loading optimizer configuration from file: {}'.format(optimizer_configuration_file_path))

            try:
                with open(optimizer_configuration_file_path, 'r') as f:
                    data = f.read()
                    optimizer_configuration = json.loads(data)
            except IOError as e:
                self.logger.log('Could not load optimizer configuration from file: {}, error: {}. Continuing without config.'.format(optimizer_configuration_file_path, e.message))
                optimizer_configuration = None

        if optimizer_name == 'adam':
            if optimizer_configuration is not None:
                optimizer = Adam.from_config(optimizer_configuration)
            else:
                lr = optimizer_info['learning_rate']
                decay = optimizer_info['decay']
                optimizer = Adam(lr=lr, decay=decay, lr_scalers=lr_scalers)

            self.logger.log('Using {} optimizer with learning rate: {}, decay: {}, beta_1: {}, beta_2: {}'
                .format(optimizer.__class__.__name__,
                        K.get_value(optimizer.lr),
                        K.get_value(optimizer.decay),
                        K.get_value(optimizer.beta_1),
                        K.get_value(optimizer.beta_2)))

        elif optimizer_name == 'sgd':
            if optimizer_configuration is not None:
                optimizer = SGD.from_config(optimizer_configuration)
            else:
                lr = optimizer_info['learning_rate']
                decay = optimizer_info['decay']
                momentum = optimizer_info['momentum']
                optimizer = SGD(lr=lr, momentum=momentum, decay=decay, lr_scalers=lr_scalers)

            self.logger.log('Using {} optimizer with learning rate: {}, momentum: {}, decay: {}'
                .format(optimizer.__class__.__name__,
                        K.get_value(optimizer.lr),
                        K.get_value(optimizer.momentum),
                        K.get_value(optimizer.decay)))

        else:
            raise ValueError('Unsupported optimizer name: {}'.format(optimizer_name))

        return optimizer

    @abstractmethod
    def _get_model_lambda_loss_type(self):
        # type: () -> ModelLambdaLossType
        pass

    @abstractmethod
    def _get_model_metrics(self):
        # type: () -> dict[str, list[Callable]]
        pass

    @abstractmethod
    def _get_model_loss(self):
        # type: () -> dict[str, Callable]
        pass

    @abstractmethod
    def _get_model_loss_weights(self):
        # type: () -> dict[str, float]
        pass

    def _get_loss_function(self):
        if self.loss_function_name == 'dummy':
            return losses.dummy_loss
        else:
            raise ValueError('Unrecognized loss function name: {}'.format(self.loss_function_name))

    def _get_compile_kwargs(self):
        if settings.PROFILE:
            return {'options': self.profiling_run_options, 'run_metadata': self.profiling_run_metadata}
        return {}

    def _get_latest_weights_file_path(self, weights_directory_path, include_early_stop=False):
        # Try to find weights from the checkpoint path
        if os.path.isdir(weights_directory_path):
            weights_folder_path = weights_directory_path
        else:
            weights_folder_path = os.path.dirname(weights_directory_path)

        weight_files = dataset_utils.get_files(weights_folder_path)

        # Filter early stop if need be
        if not include_early_stop:
            weight_files = [f for f in weight_files if 'early_stop' not in f]

        if len(weight_files) > 0:

            # Find the file with the maximum epoch index: <str>.<epoch>-<val_loss>.<str>
            max_file_idx = -1
            max_epoch = -100

            for idx, f in enumerate(weight_files):
                if os.path.isfile(os.path.join(weights_folder_path, f)) and (".hdf5" in f):
                    try:
                        epoch_and_val_loss = f.split('.')[1].rsplit('-', 1)
                        epoch = int(epoch_and_val_loss[0])

                        if epoch > max_epoch:
                            max_epoch = epoch
                            max_file_idx = idx
                    except Exception as e:
                        continue

            # Not found
            if max_file_idx < 0:
                return None

            weight_file = weight_files[max_file_idx]

            if os.path.isfile(os.path.join(weights_folder_path, weight_file)) and (".hdf5" in weight_file):
                return os.path.join(weights_folder_path, weight_file)

        return None

    def _load_latest_weights_for_model(self, model, weights_directory_path, include_early_stop=False):
        initial_epoch = 0

        try:
            self.logger.log('Searching for existing weights from checkpoint path: {}'.format(weights_directory_path))
            weight_file_path = self._get_latest_weights_file_path(weights_directory_path, include_early_stop=include_early_stop)

            if weight_file_path is None:
                self.logger.log('Could not locate any suitable weight files from the given path')
                return 0

            weight_file = weight_file_path.split('/')[-1]

            if weight_file:
                self.logger.log('Loading network weights from file: {}'.format(weight_file_path))
                model.load_weights(weight_file_path)

                # Parse the epoch number: <str>.<epoch>-<val_loss>.<str>
                epoch_val_loss = weight_file.split('.')[1].rsplit('-', 1)
                # Initial epoch is the next epoch so add one
                initial_epoch = int(epoch_val_loss[0]) + 1
                self.logger.log('Continuing training from epoch: {}'.format(initial_epoch))
            else:
                self.logger.log('No existing weights were found')

        except Exception as e:
            self.logger.log('Searching for existing weights finished with an error: {}'.format(e.message))
            return 0

        return initial_epoch

    def _transfer_weights(self, to_model_wrapper):
        # type: (ModelBase) -> dict

        transfer_weights_options = self._get_config_value('transfer_weights_options')

        if transfer_weights_options is None:
            raise ValueError('Could not find transfer weights options with key: transfer_weights_options')

        transfer_model_name = transfer_weights_options['transfer_model_name']
        transfer_model_input_shape = tuple(transfer_weights_options['transfer_model_input_shape'])
        transfer_model_num_classes = transfer_weights_options['transfer_model_num_classes']
        transfer_model_weights_file_path = transfer_weights_options['transfer_model_weights_file_path']

        self.logger.log('Creating transfer model: {} with input shape: {}, num classes: {}'
                        .format(transfer_model_name, transfer_model_input_shape, transfer_model_num_classes))
        transfer_model_wrapper = models.get_model(transfer_model_name,
                                                  transfer_model_input_shape,
                                                  transfer_model_num_classes)
        transfer_model = transfer_model_wrapper.model
        transfer_model.summary()

        self.logger.log('Loading transfer weights to transfer model from file: {}'.format(transfer_model_weights_file_path))
        transfer_model.load_weights(transfer_model_weights_file_path)

        from_layer_index = int(transfer_weights_options['from_layer_index'])
        to_layer_index = int(transfer_weights_options['to_layer_index'])
        freeze_from_layer_index = transfer_weights_options.get('freeze_from_layer_index')
        freeze_to_layer_index = transfer_weights_options.get('freeze_to_layer_index')
        scale_lr_from_layer_index = transfer_weights_options.get('scale_lr_from_layer_index')
        scale_lr_to_layer_index = transfer_weights_options.get('scale_lr_to_layer_index')
        scale_lr_factor = transfer_weights_options.get('scale_lr_factor')

        self.logger.log('Transferring weights from layer range: [{}:{}], freezing transferred layer range: [{}:{}], lr scaling layer range: [{}:{}]'
                        .format(from_layer_index, to_layer_index, freeze_from_layer_index, freeze_to_layer_index, scale_lr_from_layer_index, scale_lr_to_layer_index))

        info = to_model_wrapper.transfer_weights(
            from_model=transfer_model,
            from_layer_index=from_layer_index,
            to_layer_index=to_layer_index,
            freeze_from_layer_index=freeze_from_layer_index,
            freeze_to_layer_index=freeze_to_layer_index,
            scale_lr_from_layer_index=scale_lr_from_layer_index,
            scale_lr_to_layer_index=scale_lr_to_layer_index,
            scale_lr_factor=scale_lr_factor)

        self.logger.log('Weight transfer completed with transferred layers: {}, last transferred layer name: {}, frozen layers: {}, last frozen layer name: {}, '
                        'lr scaling layers: {}, lr scaling factor: {}'
                        .format(info.num_transferred_layers,
                                info.last_transferred_layer_name,
                                info.num_frozen_layers,
                                info.last_frozen_layer_name,
                                info.num_lr_scaling_layers,
                                scale_lr_factor))

        return info.lr_scalers

    @property
    def model_checkpoint_directory(self):
        keras_model_checkpoint = self._get_config_value('keras_model_checkpoint')
        keras_model_checkpoint_dir = self._populate_path_template(os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')))
        return keras_model_checkpoint_dir

    @property
    def model_checkpoint_file_path(self):
        keras_model_checkpoint = self._get_config_value('keras_model_checkpoint')
        keras_model_checkpoint_dir = self._populate_path_template(os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')))
        keras_model_checkpoint_file = os.path.basename(keras_model_checkpoint.get('checkpoint_file_path'))
        keras_model_checkpoint_file_path = os.path.join(keras_model_checkpoint_dir, keras_model_checkpoint_file)
        return keras_model_checkpoint_file_path

    @abstractmethod
    def train(self):
        self.logger.log('Starting training at local time {}\n'.format(datetime.datetime.now()))

        if settings.DEBUG:
            self.logger.debug_log('Training in debug mode')

        if settings.PROFILE:
            self.logger.profile_log('Training in profiling mode')

            graph_def_file_folder = self.log_folder_path
            self.logger.profile_log('Writing Tensorflow GraphDef to: {}'.format(os.path.join(graph_def_file_folder, "graph_def")))
            K.tf.train.write_graph(K.get_session().graph_def, graph_def_file_folder, "graph_def", as_text=True)
            self.logger.profile_log('Writing Tensorflow GraphDef complete')

    def modify_batch_data(self, step_index, x, y, validation=False):
        # type: (int, list, list, bool) -> (list, list)
        assert isinstance(x, list)
        assert isinstance(y, list)

        # Use a reproducible random seed to make any possible augmentations deterministic
        np.random.seed(self.random_seed + step_index)

        return x, y

    def on_batch_end(self, batch_index):
        if settings.PROFILE:
            fetched_timeline = timeline.Timeline(self.profiling_run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_dataflow=False, show_memory=True)
            self.profiling_timeliner.update_timeline(chrome_trace=chrome_trace)

    def on_epoch_end(self, epoch_index, step_index, logs):
        self.last_completed_epoch = epoch_index

    def on_training_end(self):
        if settings.PROFILE:
            self.save_profiling_timeline()

        self.logger.log('The training session ended at local time {}\n'.format(datetime.datetime.now()))
        self.logger.close_log()

    def handle_early_exit(self):
        self.logger.log('Handle early exit method called')

        # Stop training
        self.logger.log('Stopping model training')
        if self.model is not None:
            self.model.stop_training_loop()

        if not self.save_values_on_early_exit:
            self.logger.log('Save values on early exit is disabled')
            return

        # Save student model weights
        self.logger.log('Saving model weights')
        self.save_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.early-stop')

        # Save optimizer settings
        self.logger.log('Saving model optimizer settings')
        self.save_optimizer_settings(model=self.model, file_extension='.early-stop')

    def save_model_weights(self, epoch_index, val_loss, file_extension=''):
        file_path = self._populate_path_template(self.model_checkpoint_file_path, epoch=epoch_index, val_loss=val_loss) + file_extension

        # Make sure the directory exists
        general_utils.create_path_if_not_existing(file_path)

        self.logger.log('Saving model weights to file: {}'.format(file_path))
        self.model.save_weights(file_path, overwrite=True)

    def save_optimizer_settings(self, model, file_extension=''):
        file_path = self._populate_path_template(self._get_config_value('optimizer_checkpoint_file_path')) + file_extension

        with open(file_path, 'w') as log_file:
            optimizer_config = model.optimizer.get_config()
            json_str = json.dumps(optimizer_config)

            # Clear the contents of the log file
            log_file.seek(0)
            log_file.truncate()

            # Write the JSON and flush
            log_file.write(json_str)
            log_file.flush()

    def save_profiling_timeline(self):
        if settings.PROFILE:
            profiling_timeline_file_path = os.path.join(self.log_folder_path, 'profiling_timeline.json')
            self.logger.profile_log('Saving profiling data to: {}'.format(profiling_timeline_file_path))
            self.profiling_timeliner.save(profiling_timeline_file_path)


#############################################
# MEAN TEACHER TRAINER BASE
#############################################

class MeanTeacherTrainerBase(TrainerBase):

    __metaclass__ = ABCMeta

    def __init__(self, trainer_type, model_name, model_folder_name, config_file_path):
        super(MeanTeacherTrainerBase, self).__init__(trainer_type=trainer_type, model_name=model_name, model_folder_name=model_folder_name, config_file_path=config_file_path)

        # Declare instance variables
        self.teacher_model_wrapper = None
        self.teacher_validation_data_generator = None
        self.teacher_validation_data_iterator = None

        self._mean_teacher_method_config = None
        self._ema_smoothing_coefficient_function = None
        self._consistency_cost_coefficient_function = None
        self._teacher_weights_directory_path = None
        self._teacher_model_checkpoint_file_path = None

        # If we are using the mean teacher method - read the configuration and initialize the instance variables
        if self.using_mean_teacher_method:
            self.teacher_validation_data_generator = self._get_teacher_validation_data_generator()
            self.teacher_validation_data_iterator = self.teacher_validation_data_generator.get_data_set_iterator()
            self.teacher_model_wrapper = self._init_teacher_model()

        # Trigger property initializations - raise errors if properties don't exist
        if self.using_mean_teacher_method:
            assert(self.ema_smoothing_coefficient_function is not None)
            assert(self.consistency_cost_coefficient_function is not None)
            assert(self.teacher_weights_directory_path is not None)
            assert(self.teacher_model_checkpoint_file_path is not None)

    def _init_teacher_model(self):
        # If we are using the mean teacher method create the teacher model
        if self.using_mean_teacher_method:
            # Get the optimizer for the model
            optimizer = self._get_model_optimizer()
            teacher_model_lambda_loss_type = self._get_teacher_model_lambda_loss_type()

            self.logger.log('Creating teacher model {} instance with lambda loss type: {}, input shape: {}, num classes: {}'.format(self.model_name, teacher_model_lambda_loss_type, self.input_shape, self.num_classes))
            self.teacher_model_wrapper = models.get_model(self.model_name, self.input_shape, self.num_classes, logger=self.logger, model_lambda_loss_type=teacher_model_lambda_loss_type)
            self.teacher_model.summary()

            if self.continue_from_last_checkpoint:
                self.logger.log('Loading latest teacher model weights from path: {}'.format(self.teacher_weights_directory_path))
                initial_teacher_epoch = self._load_latest_weights_for_model(self.teacher_model, self.teacher_weights_directory_path)

                if initial_teacher_epoch < 1:
                    self.logger.warn('Could not find suitable weights, initializing teacher with student model weights')
                    self.teacher_model.set_weights(self.model.get_weights())
            else:
                self.teacher_model.set_weights(self.model.get_weights())

            teacher_model_loss = self._get_teacher_model_loss()
            teacher_model_metrics = self._get_teacher_model_metrics()

            self.teacher_model.compile(optimizer=optimizer,
                                       loss=teacher_model_loss,
                                       metrics=teacher_model_metrics,
                                       **self._get_compile_kwargs())

            # Log the model structure to a file using the keras plot_model
            try:
                model_plot_file_path = os.path.join(self.log_folder_path, 'teacher_model.png')
                self.logger.log('Saving teacher model plot to file: {}'.format(model_plot_file_path))
                plot_model(self.teacher_model, to_file=model_plot_file_path, show_shapes=True, show_layer_names=True)
            except Exception as e:
                self.logger.warn('Saving teacher model plot to file failed: {}'.format(e.message))

            return self.teacher_model_wrapper

        return None

    @abstractmethod
    def _get_teacher_validation_data_generator(self):
        pass

    @property
    def teacher_model(self):
        if self.using_mean_teacher_method and self.teacher_model_wrapper is not None:
            return self.teacher_model_wrapper.model

        return None

    @property
    def mean_teacher_method_config(self):
        if self.using_mean_teacher_method and self._mean_teacher_method_config is None:
            self.logger.log('Reading mean teacher method configuration')
            self._mean_teacher_method_config = self._get_config_value('mean_teacher_params')

            if self._mean_teacher_method_config is None:
                raise ValueError('Could not find entry for mean_teacher_params from the configuration JSON')

        return self._mean_teacher_method_config

    @property
    def ema_smoothing_coefficient_function(self):
        if self.using_mean_teacher_method and self._ema_smoothing_coefficient_function is None:
            ema_coefficient_schedule_function = self.mean_teacher_method_config['ema_smoothing_coefficient_function']
            self.logger.log('Mean teacher EMA smoothing coefficient function: {}'.format(ema_coefficient_schedule_function))
            self._ema_smoothing_coefficient_function = eval(ema_coefficient_schedule_function)

        return self._ema_smoothing_coefficient_function

    @property
    def consistency_cost_coefficient_function(self):
        if self.using_mean_teacher_method and self._consistency_cost_coefficient_function is None:
            consistency_cost_coefficient_function = self.mean_teacher_method_config['consistency_cost_coefficient_function']
            self.logger.log('Mean teacher consistency cost coefficient function: {}'.format(consistency_cost_coefficient_function))
            self._consistency_cost_coefficient_function = eval(consistency_cost_coefficient_function)

        return self._consistency_cost_coefficient_function

    @property
    def teacher_weights_directory_path(self):
        if self.using_mean_teacher_method and self._teacher_weights_directory_path is None:
            self._teacher_weights_directory_path = self._populate_path_template(os.path.dirname(self.mean_teacher_method_config['teacher_model_checkpoint_file_path']))
            self.logger.log('Teacher weights directory path: {}'.format(self._teacher_weights_directory_path))

        return self._teacher_weights_directory_path

    @property
    def teacher_model_checkpoint_file_path(self):
        if self.using_mean_teacher_method and self._teacher_model_checkpoint_file_path is None:
            self._teacher_model_checkpoint_file_path= self.mean_teacher_method_config['teacher_model_checkpoint_file_path']
            self.logger.log('Teacher checkpoint file path: {}'.format(self._teacher_model_checkpoint_file_path))

        return self._teacher_model_checkpoint_file_path

    @abstractproperty
    def using_mean_teacher_method(self):
        pass

    @abstractmethod
    def _get_teacher_model_lambda_loss_type(self):
        # type: () -> ModelLambdaLossType
        pass

    @abstractmethod
    def _get_teacher_model_loss(self):
        # type: () -> Callable
        pass

    @abstractmethod
    def _get_teacher_model_metrics(self):
        # type: () -> list[Callable]
        pass

    def _save_teacher_model_weights(self, epoch_index, val_loss, file_extension='.teacher'):
        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot save weights')

            # Save the teacher model weights:
            teacher_model_checkpoint_file_path = self.teacher_model_checkpoint_file_path

            # Don't crash here, too much effort done - save with a different name to the same path as
            # the student model
            if teacher_model_checkpoint_file_path is None:
                self.logger.log('Value of teacher_model_checkpoint_file_path is not set - defaulting to teacher folder under student directory')
                file_name_format = os.path.basename(self.model_checkpoint_file_path)
                teacher_model_checkpoint_file_path = os.path.join(os.path.join(self.model_checkpoint_directory, 'teacher/'), file_name_format)

            file_path = self._populate_path_template(teacher_model_checkpoint_file_path).format(epoch=epoch_index, val_loss=val_loss) + file_extension

            # Make sure the directory exists
            general_utils.create_path_if_not_existing(file_path)

            self.logger.log('Saving mean teacher model weights to file: {}'.format(file_path))
            self.teacher_model.save_weights(file_path, overwrite=True)

    def modify_batch_data(self, step_index, x, y, validation=False):
        # type: (int, list[np.ndarray[np.float32]], np.array, bool) -> (list[np.ndarray[np.float32]], np.array)

        """
        Invoked by the ExtendedModel right before train_on_batch:

        If using the Mean Teacher method:
            Modifies the batch data by appending the mean teacher predictions as the last
            element of the input data X if we are using mean teacher training.

        # Arguments
            :param step_index: the training step index
            :param x: input data
            :param y: output data
            :param validation: is this a validation data batch
        # Returns
            :return: a tuple of (input data, output data)
        """
        x, y = super(MeanTeacherTrainerBase, self).modify_batch_data(step_index=step_index, x=x, y=y, validation=validation)

        # Append mean teacher data
        if self.using_mean_teacher_method:
            img_batch = x[0]
            labels_data = x[1]

            # Parse the teacher data shape, this should work for classification and segmentation regardless of the
            # label encoding as the model predictions should always yield logits for each class and the batch dimension
            # should always be the first dimension in any input
            # TODO: Fix to work in a more general case
            if self.trainer_type == TrainerType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER or self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED_MEAN_TEACHER:
                teacher_data_shape = list(labels_data.shape)
            else:
                teacher_data_shape = list(img_batch.shape[0:-1]) + [self.num_classes]

            # Teacher data is supposed to be the last input in the input data
            # pop that from the input data and run evaluation
            if not validation:
                teacher_img_batch = x[-1]
                x = x[:-1]
            else:
                teacher_img_batch = img_batch

            x = x + self._get_mean_teacher_extra_batch_data(teacher_img_batch, step_index=step_index, teacher_data_shape=teacher_data_shape, validation=validation)

        return x, y

    def on_batch_end(self, step_index):
        # type: (int) -> ()

        """
        Invoked by the ExtendedModel right after train_on_batch:

        Updates the teacher model weights if using the mean teacher method for
        training, otherwise does nothing.

        # Arguments
            :param step_index: the training step index
        # Returns
            Nothing
        """

        super(MeanTeacherTrainerBase, self).on_batch_end(step_index)

        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot run EMA update')

            a = self.ema_smoothing_coefficient_function(step_index)

            if not 0 <= a <= 1.0:
                self.logger.warn('Out of bounds EMA coefficient value when updating teacher weights: {}'.format(a))

            # Perform the EMA weight update: theta'_t = a * theta'_t-1 + (1 - a) * theta_t
            t_weights = self.teacher_model.get_weights()
            s_weights = self.model.get_weights()

            if len(t_weights) != len(s_weights):
                raise ValueError('The weight arrays are not of the same length for the student and teacher: {} vs {}'
                                 .format(len(t_weights), len(s_weights)))

            num_weights = len(t_weights)
            s_time = time.time()

            for i in range(0, num_weights):
                t_weights[i] = a * t_weights[i] + ((1.0 - a) * s_weights[i])

            self.teacher_model.set_weights(t_weights)
            self.logger.profile_log('Mean teacher weight update took: {} s'.format(time.time() - s_time))

    def on_epoch_end(self, epoch_index, step_index, logs):
        # type: (int, int, dict) -> ()

        """
        Invoked by the ExtendedModel right after the epoch is over.

        Evaluates mean teacher model on the validation data and saves the mean teacher
        model weights.

        # Arguments
            :param epoch_index: index of the epoch that has finished
            :param step_index: index of the step that has finished
            :param logs: logs from the epoch (for the student model)
        # Returns
            Nothing
        """
        super(MeanTeacherTrainerBase, self).on_epoch_end(epoch_index, step_index, logs)

        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot validate/save weights')

            # Default to -1.0 validation loss if nothing else is given
            val_loss = -1.0

            if self.teacher_validation_data_generator is not None and self.teacher_validation_data_iterator is not None:
                # Evaluate the mean teacher on the validation data
                validation_steps_per_epoch = self.teacher_validation_data_iterator.num_steps_per_epoch

                val_outs = self.teacher_model.evaluate_generator(
                    generator=self.teacher_validation_data_iterator,
                    steps=validation_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH,
                    workers=max(dataset_utils.get_number_of_parallel_jobs()-1, 1),
                    use_multiprocessing=True,
                    validation=True)

                # Parse all validation metrics to a single string
                metrics_names = self.teacher_model.metrics_names
                val_outs_str = ""

                for i in range(0, len(val_outs)):
                    val_outs_str = val_outs_str + "val_{}: {}, ".format(metrics_names[i], val_outs[i])

                    if metrics_names[i] == 'loss':
                        val_loss = val_outs[i]

                val_outs_str = val_outs_str[0:-2]

                self.logger.log('Epoch {}: Teacher model {}'.format(epoch_index, val_outs_str))

            self.logger.log('Epoch {}: EMA coefficient {}, consistency cost coefficient: {}'
                            .format(epoch_index, self.ema_smoothing_coefficient_function(step_index),
                                    self.consistency_cost_coefficient_function(step_index)))
            self.save_teacher_model_weights(epoch_index=epoch_index, val_loss=val_loss)

    def save_teacher_model_weights(self, epoch_index, val_loss, file_extension=''):
        # type: (int, float, str) -> None

        """
        # Arguments
        :param epoch_index: Index of the epoch (encoded in the file name)
        :param val_loss: Validation loss (encoded in the file name)
        :param file_extension: Optional extension for the file

        # Returns
            None
        """
        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot save teacher model weights')

            # Save the teacher model weights:
            teacher_model_checkpoint_file_path = self.teacher_model_checkpoint_file_path

            # Don't crash here, too much effort done - save with a different name to the same path as
            # the student model
            if teacher_model_checkpoint_file_path is None:
                self.logger.log('Value of teacher_model_checkpoint_file_path is not set - defaulting to teacher folder under student directory')
                file_name_format = os.path.basename(self.model_checkpoint_file_path)
                teacher_model_checkpoint_file_path = os.path.join(os.path.join(self.model_checkpoint_directory, 'teacher/'), file_name_format)

            file_path = self._populate_path_template(teacher_model_checkpoint_file_path, epoch=epoch_index, val_loss=val_loss) + file_extension

            # Make sure the directory exists
            general_utils.create_path_if_not_existing(file_path)

            self.logger.log('Saving mean teacher model weights to file: {}'.format(file_path))
            self.teacher_model.save_weights(file_path, overwrite=True)

    def handle_early_exit(self):
        super(MeanTeacherTrainerBase, self).handle_early_exit()

        if not self.save_values_on_early_exit:
            return

        # Save teacher model weights
        if self.using_mean_teacher_method:
            if self.teacher_model is not None:
                self.logger.log('Saving teacher model weights')
                self.save_teacher_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.early-stop')

    def _get_mean_teacher_extra_batch_data(self, teacher_img_batch, step_index, teacher_data_shape, validation):
        # type: (np.ndarray, int, list, bool) -> list

        """
        Calculates the extra batch data necessary for the Mean Teacher method. In other words returns a list with
        two objects: mean teacher predictions for the image batch and consistency coefficients. The first dimension
        in both is equal to the batch dimension.

        If it's a validation round the data will be dummy data i.e. zeros

        # Arguments
            :param teacher_img_batch: The input images to the teacher neural network
            :param step_index: Step index (used in coefficient calculation)
            :param teacher_data_shape: Shape of the teacher data - might not always be the same as img batch e.g. for classification
            :param validation: True if this is a validation batch false otherwise
        # Returns
            :return: The Mean Teacher extra data as a list [mt_predictions, consistency_coefficients]
        """

        if not self.using_mean_teacher_method:
            return []

        if self.teacher_model is None:
            raise ValueError('Teacher model is not set, cannot run predictions')

        # First dimension in all of the input data should be the batch size
        batch_size = teacher_img_batch.shape[0]

        if validation:
            # BxHxWxC
            mean_teacher_predictions = np.zeros(shape=teacher_data_shape)
            np_consistency_coefficients = np.zeros(shape=[batch_size])
            return [mean_teacher_predictions, np_consistency_coefficients]
        else:
            # Note: include the training phase noise and dropout layers on the prediction
            s_time = time.time()
            mean_teacher_predictions = self.teacher_model.predict_on_batch(teacher_img_batch, use_training_phase_layers=True)
            self.logger.profile_log('Mean teacher batch predictions took: {} s'.format(time.time() - s_time))
            consistency_coefficient = self.consistency_cost_coefficient_function(step_index)
            np_consistency_coefficients = np.ones(shape=[batch_size]) * consistency_coefficient

            if teacher_data_shape != list(mean_teacher_predictions.shape):
                self.logger.warn('Mismatch between teacher data shape and returned MT predictions: {} vs {}'
                                 .format(teacher_data_shape, mean_teacher_predictions.shape))

            return [mean_teacher_predictions, np_consistency_coefficients]


#############################################
# SEGMENTATION TRAINER
#############################################


class SegmentationTrainer(MeanTeacherTrainerBase):

    def __init__(self,
                 trainer_type,
                 model_name,
                 model_folder_name,
                 config_file_path):
        # type: (str, str, str, str, str) -> ()

        # Declare instance variables
        self._label_generation_function = None
        self._data_set_information = None
        self._material_class_information = None
        self._class_weights = None
        self._ignore_classes = None
        self._training_data_generator_params = None
        self._validation_data_generator_params = None
        self._superpixel_params = None
        self._superpixel_label_generation_function_type = SuperpixelSegmentationFunctionType.NONE
        self._superpixel_unlabeled_cost_coefficient_function = None

        super(SegmentationTrainer, self).__init__(trainer_type=trainer_type, model_name=model_name, model_folder_name=model_folder_name, config_file_path=config_file_path)

        # Trigger property initializations - raise errors if don't exist
        if self.using_superpixel_method:
            assert(self.superpixel_label_generation_function_type is not None)
            assert(self.superpixel_unlabeled_cost_coefficient_function is not None)

    """
    PROPERTIES
    """

    @property
    def superpixel_method_config(self):
        if self.using_superpixel_method and self._superpixel_params is None:
            self.logger.log('Reading superpixel method configuration')
            self._superpixel_params = self._get_config_value('superpixel_params')

            if self._superpixel_params is None:
                raise ValueError('Could not find entry for superpixel_params from the configuration JSON')

        return self._superpixel_params

    @property
    def superpixel_label_generation_function_type(self):
        if self.using_superpixel_method and self._superpixel_label_generation_function_type == SuperpixelSegmentationFunctionType.NONE:
            label_generation_function_name = self.superpixel_method_config['label_generation_function_name']
            self.logger.log('Superpixel label generation function name: {}'.format(label_generation_function_name))
            self._superpixel_label_generation_function_type = self._get_superpixel_label_generation_function_type(label_generation_function_name)

        return self._superpixel_label_generation_function_type

    @property
    def superpixel_unlabeled_cost_coefficient_function(self):
        if self.using_superpixel_method and self._superpixel_unlabeled_cost_coefficient_function is None:
            unlabeled_cost_coefficient_function = self.superpixel_method_config['unlabeled_cost_coefficient_function']
            self.logger.log('Superpixel unlabeled cost coefficient function: {}'.format(unlabeled_cost_coefficient_function))
            self._superpixel_unlabeled_cost_coefficient_function = eval(unlabeled_cost_coefficient_function)

        return self._superpixel_unlabeled_cost_coefficient_function

    @property
    def num_unlabeled_per_batch(self):
        num_unlabeled_per_batch = int(self._get_config_value('num_unlabeled_per_batch'))

        # Sanity check for number of unlabeled per batch
        if num_unlabeled_per_batch > 0 and self.is_supervised_only_trainer:
            self.logger.warn('Trainer type is marked as {} with unlabeled per batch {} - assuming 0 unlabeled per batch'.format(self.trainer_type, num_unlabeled_per_batch))
            return 0

        return num_unlabeled_per_batch

    @property
    def path_to_material_class_file(self):
        return self._get_config_value('path_to_material_class_file')

    @property
    def path_to_data_set_information_file(self):
        return self._get_config_value('path_to_data_set_information_file')

    @property
    def path_to_labeled_photos(self):
        return self._get_config_value('path_to_labeled_photos')

    @property
    def path_to_labeled_masks(self):
        return self._get_config_value('path_to_labeled_masks')

    @property
    def path_to_unlabeled_photos(self):
        return self._get_config_value('path_to_unlabeled_photos')

    @property
    def data_set_information(self):
        # Lazy load data set information
        if self._data_set_information is None:
            self.logger.log('Loading data set information from: {}'.format(self.path_to_data_set_information_file))
            self._data_set_information = dataset_utils.load_segmentation_data_set_information(self.path_to_data_set_information_file)
            self.logger.log('Loaded data set information successfully with set sizes (tr, va, te): {}, {}, {}'
                            .format(self.data_set_information.training_set.labeled_size + self.data_set_information.training_set.unlabeled_size,
                                    self.data_set_information.validation_set.labeled_size,
                                    self.data_set_information.test_set.labeled_size))

        return self._data_set_information

    @property
    def material_class_information(self):
        # Lazy load material class information
        if self._material_class_information is None:
            self.logger.log('Loading material class information from: {}'.format(self.path_to_material_class_file))
            self._material_class_information = dataset_utils.load_material_class_information(self.path_to_material_class_file)
            self.logger.log('Loaded {} material classes successfully'.format(self.num_classes))

        return self._material_class_information

    @property
    def class_weights(self):
        # Lazy load class weights when needed
        if self._class_weights is None:
            if self.use_class_weights:
                class_weights = self.data_set_information.class_weights
                override_class_weights = self._get_config_value('override_class_weights')

                # Legacy support for data sets without material_samples_class_weights
                if self.using_material_samples and hasattr(self.data_set_information, 'material_samples_class_weights'):
                    if self.data_set_information.material_samples_class_weights is not None and len(self.data_set_information.material_samples_class_weights) > 0:
                        class_weights = self.data_set_information.material_samples_class_weights
                    else:
                        self.logger.warn('The trainer is using material samples but no material_samples_class_weights '
                                         'could be found - use override class weights or recreate the dataset.')
                elif self.using_material_samples and not hasattr(self.data_set_information, 'material_samples_class_weights'):
                    self.logger.warn('The trainer is using material samples but no material_samples_class_weights '
                                     'could be found - use override class weights or recreate the dataset.')

                if override_class_weights is not None:
                    self.logger.log('Found override class weights: {}'.format(override_class_weights))
                    self.logger.log('Using override class weights instead of data set information class weights')
                    class_weights = override_class_weights

                if class_weights is None:
                    raise ValueError('Existing class weights were not found')
                if len(class_weights) != self.num_classes:
                    raise ValueError('Number of classes in class weights did not match number of material classes: {} vs {}'
                                     .format(len(class_weights), self.num_classes))

                self.logger.log('Using class weights: {}'.format(class_weights))
                class_weights = np.array(class_weights, dtype=np.float32)
                self._class_weights = class_weights
            else:
                self._class_weights = np.ones([self.num_classes], dtype=np.float32)

        return self._class_weights

    @property
    def ignore_classes(self):
        if self._ignore_classes is None:
            ignore_classes = np.squeeze(np.where(np.equal(self.class_weights, 0.0))).astype(dtype=np.int32)

            # If there is only a single ignored class the shape will be empty - expand to dim 1
            if ignore_classes.shape == ():
                ignore_classes = np.expand_dims(ignore_classes, axis=-1)

            self._ignore_classes = ignore_classes

        return self._ignore_classes

    @property
    def training_data_generator_params(self):
        if self._training_data_generator_params is None:
            # Note: the initial epoch is an index but the property is a number starting from 1
            self._training_data_generator_params = SegmentationDataGeneratorParameters(
                material_class_information=self.material_class_information,
                num_color_channels=self.num_color_channels,
                num_crop_reattempts=self.num_crop_reattempts,
                random_seed=self.random_seed,
                crop_shapes=self.crop_shape,
                resize_shapes=self.resize_shape,
                use_per_channel_mean_normalization=True,
                per_channel_mean=self.per_channel_mean,
                use_per_channel_stddev_normalization=True,
                per_channel_stddev=self.per_channel_stddev,
                use_data_augmentation=self.use_data_augmentation,
                use_material_samples=self.using_material_samples,
                use_selective_attention=self.using_selective_attention,
                use_adaptive_sampling=self.using_adaptive_sampling,
                data_augmentation_params=self.data_augmentation_parameters,
                shuffle_data_after_epoch=True,
                div2_constraint=self.div2_constraint,
                initial_epoch=self.initial_epoch)

        return self._training_data_generator_params

    @property
    def validation_data_generator_params(self):
        if self._validation_data_generator_params is None:
            self._validation_data_generator_params = SegmentationDataGeneratorParameters(
                material_class_information=self.material_class_information,
                num_color_channels=self.num_color_channels,
                num_crop_reattempts=self.num_crop_reattempts,
                random_seed=self.random_seed,
                crop_shapes=self.validation_crop_shape,
                resize_shapes=self.validation_resize_shape,
                use_per_channel_mean_normalization=True,
                per_channel_mean=self.per_channel_mean,
                use_per_channel_stddev_normalization=True,
                per_channel_stddev=self.per_channel_stddev,
                use_data_augmentation=False,
                use_material_samples=False,
                use_selective_attention=False,
                use_adaptive_sampling=False,
                data_augmentation_params=None,
                shuffle_data_after_epoch=True,
                div2_constraint=self.div2_constraint)

        return self._validation_data_generator_params

    @property
    def num_classes(self):
        return len(self.material_class_information)

    @property
    def num_crop_reattempts(self):
        val = self._get_config_value('num_crop_reattempts')
        return int(val) if val is not None else 0

    @property
    def is_supervised_only_trainer(self):
        return self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED or \
               self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED_MEAN_TEACHER

    @property
    def using_material_samples(self):
        return bool(self._get_config_value('use_material_samples'))

    @property
    def using_selective_attention(self):
        return bool(self._get_config_value('use_selective_attention'))

    @property
    def using_adaptive_sampling(self):
        return bool(self._get_config_value('use_adaptive_sampling'))

    @property
    def using_mean_teacher_method(self):
        return self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED_MEAN_TEACHER or \
               self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER or \
               self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL

    @property
    def per_channel_mean(self):
        # type: () -> np.ndarray

        if self.using_unlabeled_training_data:
            return np.array(self.data_set_information.per_channel_mean, dtype=np.float32)
        else:
            return np.array(self.data_set_information.labeled_per_channel_mean, dtype=np.float32)

    @property
    def per_channel_stddev(self):
        # type: () -> np.ndarray

        if self.using_unlabeled_training_data:
            return np.array(self.data_set_information.per_channel_stddev, dtype=np.float32)
        else:
            return np.array(self.data_set_information.labeled_per_channel_stddev, dtype=np.float32)

    @property
    def using_superpixel_method(self):
        return self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL or \
               self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL

    @property
    def using_unlabeled_training_data(self):
        return not self.is_supervised_only_trainer and \
               self.training_set_unlabeled is not None and \
               self.training_set_unlabeled.size > 0 and \
               self.num_unlabeled_per_batch > 0

    @property
    def total_batch_size(self):
        if not self.using_unlabeled_training_data:
            return self.num_labeled_per_batch

        return self.num_labeled_per_batch + self.num_unlabeled_per_batch

    def _get_data_sets(self):
        self.logger.log('Initializing data')

        self.logger.log('Creating labeled data sets with photo files from: {} and mask files from: {}'
                        .format(self.path_to_labeled_photos, self.path_to_labeled_masks))

        # Labeled training set
        self.logger.log('Creating labeled training set')
        stime = time.time()
        training_set_labeled = LabeledImageDataSet('training_set_labeled',
                                                   path_to_photo_archive=self.path_to_labeled_photos,
                                                   path_to_mask_archive=self.path_to_labeled_masks,
                                                   photo_file_list=self.data_set_information.training_set.labeled_photos,
                                                   mask_file_list=self.data_set_information.training_set.labeled_masks,
                                                   material_samples=self.data_set_information.training_set.material_samples)
        self.logger.log('Labeled training set construction took: {} s, size: {}'.format(time.time()-stime, training_set_labeled.size))

        if training_set_labeled.size == 0:
            raise ValueError('No training data found')

        # Unlabeled training set - skip construction if unlabeled per batch is zero or this is a supervised segmentation trainer
        if self.num_unlabeled_per_batch > 0 and not self.is_supervised_only_trainer:
            self.logger.log('Creating unlabeled training set from: {}'.format(self.path_to_unlabeled_photos))
            stime = time.time()
            training_set_unlabeled = UnlabeledImageDataSet('training_set_unlabeled',
                                                           path_to_photo_archive=self.path_to_unlabeled_photos,
                                                           photo_file_list=self.data_set_information.training_set.unlabeled_photos)
            self.logger.log('Unlabeled training set creation took: {} s, size: {}'.format(time.time()-stime, training_set_unlabeled.size))
        else:
            training_set_unlabeled = None

        # Labeled validation set
        self.logger.log('Creating validation set')
        stime = time.time()
        validation_set = LabeledImageDataSet('validation_set',
                                             path_to_photo_archive=self.path_to_labeled_photos,
                                             path_to_mask_archive=self.path_to_labeled_masks,
                                             photo_file_list=self.data_set_information.validation_set.labeled_photos,
                                             mask_file_list=self.data_set_information.validation_set.labeled_masks,
                                             material_samples=self.data_set_information.validation_set.material_samples)
        self.logger.log('Labeled validation set creation took: {} s, size: {}'.format(time.time()-stime, validation_set.size))

        # Labeled test set
        #self.logger.log('Creating test set')
        #stime = time.time()
        #test_set = LabeledImageDataSet('test_set',
        #                                    path_to_photo_archive=self.path_to_labeled_photos,
        #                                    path_to_mask_archive=self.path_to_labeled_masks,
        #                                    photo_file_list=self.data_set_information.test_set.labeled_photos,
        #                                    mask_file_list=self.data_set_information.test_set.labeled_masks,
        #                                    material_samples=self.data_set_information.test_set.material_samples)
        #self.logger.log('Labeled test set creation took: {} s, size: {}'.format(time.time()-stime, test_set.size))

        if training_set_unlabeled is not None:
            total_data_set_size = training_set_labeled.size + training_set_unlabeled.size + validation_set.size
        else:
            total_data_set_size = training_set_labeled.size + validation_set.size

        self.logger.log('Total data set size (training + val): {}'.format(total_data_set_size))

        return training_set_labeled, training_set_unlabeled, validation_set

    def _get_data_generators(self):
        self.logger.log('Initializing data generators')

        # Create training data and validation data generators
        # Note: training data comes from semi-supervised segmentation data generator and validation
        # and test data come from regular segmentation data generator
        self.logger.log('Creating training data generator')

        training_data_generator = SegmentationDataGenerator(
            labeled_data_set=self.training_set_labeled,
            unlabeled_data_set=self.training_set_unlabeled if self.using_unlabeled_training_data else None,
            num_labeled_per_batch=self.num_labeled_per_batch,
            num_unlabeled_per_batch=self.num_unlabeled_per_batch if self.using_unlabeled_training_data else 0,
            params=self.training_data_generator_params,
            class_weights=self.class_weights,
            batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
            label_generation_function_type=self.superpixel_label_generation_function_type)

        self.logger.log('Creating validation data generator')

        # The student lambda loss layer needs semi-supervised input, so we need to work around it
        # to only provide labeled input from the semi-supervised data generator. The dummy data
        # is appended to each batch so that the batch data maintains it's shape. This is done in the
        # modify_batch_data function.
        validation_data_generator = SegmentationDataGenerator(
            labeled_data_set=self.validation_set,
            unlabeled_data_set=None,
            num_labeled_per_batch=self.validation_num_labeled_per_batch,
            num_unlabeled_per_batch=0,
            params=self.validation_data_generator_params,
            class_weights=self.class_weights,
            batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
            label_generation_function_type=SuperpixelSegmentationFunctionType.NONE)

        self.logger.log('Using unlabeled training data: {}'.format(self.using_unlabeled_training_data))
        self.logger.log('Using material samples: {}'.format(training_data_generator.use_material_samples))
        self.logger.log('Using per-channel mean: {}'.format(training_data_generator.per_channel_mean))
        self.logger.log('Using per-channel stddev: {}'.format(training_data_generator.per_channel_stddev))

        return training_data_generator, validation_data_generator

    def _get_teacher_validation_data_generator(self):
        # Note: The teacher has a supervised batch data format for validation data generation
        # because it doesn't have the semi-supervised loss lambda layer since we need to predict with it
        if self.using_mean_teacher_method:
            self.logger.log('Creating teacher validation data generator')

            teacher_validation_data_generator = SegmentationDataGenerator(
                labeled_data_set=self.validation_set,
                unlabeled_data_set=None,
                num_labeled_per_batch=self.validation_num_labeled_per_batch,
                num_unlabeled_per_batch=0,
                params=self.validation_data_generator_params,
                class_weights=self.class_weights,
                batch_data_format=BatchDataFormat.SUPERVISED,
                label_generation_function_type=SuperpixelSegmentationFunctionType.NONE)
        else:
            teacher_validation_data_generator = None

        return teacher_validation_data_generator

    def train(self):
        # type: () -> History
        super(SegmentationTrainer, self).train()

        assert isinstance(self.model, ExtendedModel)

        # Labeled data set size determines the epochs
        num_workers = max(dataset_utils.get_number_of_parallel_jobs()-1, 1)

        if self.using_unlabeled_training_data:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}, unlabeled data set size: {}, num unlabeled per batch: {}'
                            .format(self.training_set_labeled.size, self.num_labeled_per_batch, self.training_set_unlabeled.size, self.num_unlabeled_per_batch))
        else:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}'
                            .format(self.training_set_labeled.size, self.num_labeled_per_batch))

        self.logger.log('Num epochs: {}, initial epoch: {}, total batch size: {}, crop shape: {}, training steps per epoch: {}, validation steps per epoch: {}'
                        .format(self.num_epochs, self.initial_epoch, self.total_batch_size, self.crop_shape, self.training_steps_per_epoch, self.validation_steps_per_epoch))

        self.logger.log('Num workers: {}'.format(num_workers))

        # Get a list of callbacks
        callbacks = self._get_training_callbacks()

        # Note: the student model should not be evaluated using the validation data generator
        # the generator input will not be meaning
        history = self.model.fit_generator(
            generator=self.training_data_iterator,
            steps_per_epoch=self.training_steps_per_epoch,
            epochs=self.num_epochs,
            initial_epoch=self.initial_epoch,
            validation_data=self.validation_data_iterator,
            validation_steps=self.validation_steps_per_epoch,
            verbose=1,
            trainer=self,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=num_workers)

        return history

    def handle_early_exit(self):
        super(SegmentationTrainer, self).handle_early_exit()
        self.logger.log('Early exit handler complete - ready for exit')
        self.logger.close_log()

    def modify_batch_data(self, step_index, x, y, validation=False):
        # type: (int, list[np.array[np.float32]], np.array, bool) -> (list[np.array[np.float32]], np.array)

        """
        Invoked by the ExtendedModel right before train_on_batch:

        If using the Mean Teacher method:
            Modifies the batch data by appending the mean teacher predictions as the last
            element of the input data X if we are using mean teacher training.

        If using superpixel semi-supervised:
            Modifies the batch data by appending the unlabeled data cost coefficients.

        If using both:
            Appends first mean teacher then superpixel extra data

        # Arguments
            :param step_index: the training step index
            :param x: input data
            :param y: output data
            :param validation: is this a validation data batch
        # Returns
            :return: a tuple of (input data, output data)
        """
        x, y = super(SegmentationTrainer, self).modify_batch_data(step_index=step_index, x=x, y=y, validation=validation)

        img_batch = x[0]
        mask_batch = x[1]

        # Append first mean teacher data and then superpixel data if using both methods
        # mean teacher data is handled by the MeanTeacherTrainerBase - base class
        if self.using_superpixel_method:
            x = x + self._get_superpixel_extra_batch_data(img_batch, step_index=step_index, validation=validation)

        return x, y

    def _get_model_lambda_loss_type(self):
        # type: () -> ModelLambdaLossType

        if self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED:
            return ModelLambdaLossType.SEGMENTATION_CATEGORICAL_CROSS_ENTROPY
        if self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED_MEAN_TEACHER:
            return ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER            # Same lambda loss as semi-supervised but only with 0 unlabeled data
        elif self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER:
            return ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER
        elif self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL:
            return ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL
        elif self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL:
            return ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL
        else:
            raise ValueError('Unsupported combination for a semisupervised trainer - cannot deduce model type')

    def _get_model_loss(self):
        return {'loss': losses.dummy_loss, 'logits': lambda _, y_pred: 0.0*y_pred}

    def _get_model_loss_weights(self):
        return {'loss': 1., 'logits': 0.}

    def _get_model_metrics(self):
        return {'logits': [metrics.segmentation_accuracy(self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                           metrics.segmentation_mean_iou(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                           metrics.segmentation_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes)]}

    def _get_teacher_model_lambda_loss_type(self):
        return ModelLambdaLossType.NONE

    def _get_teacher_model_loss(self):
        return losses.segmentation_sparse_weighted_categorical_cross_entropy(self.class_weights)

    def _get_teacher_model_metrics(self):
        return [metrics.segmentation_accuracy(self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                metrics.segmentation_mean_iou(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                metrics.segmentation_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes)]

    def _get_superpixel_extra_batch_data(self, img_batch, step_index, validation):
        if not self.using_superpixel_method:
            return []

        # First dimension in all of the input data should be the batch size
        batch_size = img_batch.shape[0]

        if validation:
            np_unlabeled_cost_coefficients = np.zeros(shape=[batch_size])
            return [np_unlabeled_cost_coefficients]
        else:
            unlabeled_cost_coefficient = self.superpixel_unlabeled_cost_coefficient_function(step_index)
            np_unlabeled_cost_coefficients = np.ones(shape=[batch_size]) * unlabeled_cost_coefficient
            return [np_unlabeled_cost_coefficients]

    def _get_superpixel_label_generation_function_type(self, label_generation_function_name):
        # type: (str) -> SuperpixelSegmentationFunctionType
        function_name = label_generation_function_name.lower()

        if function_name == 'felzenswalb':
            return SuperpixelSegmentationFunctionType.FELZENSWALB
        elif function_name == 'slic':
            return SuperpixelSegmentationFunctionType.SLIC
        elif function_name == 'watershed':
            return SuperpixelSegmentationFunctionType.WATERSHED
        elif function_name == 'quickshift':
            return SuperpixelSegmentationFunctionType.QUICKSHIFT
        else:
            raise ValueError('Invalid superpixel label generation function name: {}'.format(function_name))

#############################################
# CLASSIFICATION TRAINER
#############################################

class ClassificationTrainer(MeanTeacherTrainerBase):

    def __init__(self,
                 trainer_type,
                 model_name,
                 model_folder_name,
                 config_file_path):

        # Declare instance variables
        self._classification_data_set_config = None
        self._ignore_classes = None
        self._class_weights = None

        self._training_data_generator_params = None
        self._validation_data_generator_params = None

        super(ClassificationTrainer, self).__init__(trainer_type=trainer_type,
                                                    model_name=model_name,
                                                    model_folder_name=model_folder_name,
                                                    config_file_path=config_file_path)

    @property
    def path_to_labeled_photos(self):
        return self.config['path_to_labeled_photos']

    @property
    def path_to_unlabeled_photos(self):
        return self.config['path_to_unlabeled_photos']

    @property
    def classification_data_set_config(self):
        if self._classification_data_set_config is None:
            self.logger.log('Reading classification data set configuration')
            self._classification_data_set_config = self._get_config_value('classification_data_set_params')

            if self._classification_data_set_config is None:
                raise ValueError('Could not find entry for classification_data_set_params from the configuration JSON')

        return self._classification_data_set_config

    @property
    def path_to_label_mapping_file(self):
        return self.classification_data_set_config['path_to_label_mapping_file']

    @property
    def path_to_training_set_file(self):
        return self.classification_data_set_config['path_to_training_set_file']

    @property
    def path_to_validation_set_file(self):
        return self.classification_data_set_config['path_to_validation_set_file']

    @property
    def path_to_test_set_file(self):
        return self.classification_data_set_config['path_to_test_set_file']

    @property
    def per_channel_mean(self):
        # type: () -> np.ndarray

        if self.using_unlabeled_training_data:
            return np.array(self.classification_data_set_config['per_channel_mean'], dtype=np.float32)
        else:
            return np.array(self.classification_data_set_config['per_channel_mean_labeled'], dtype=np.float32)

    @property
    def per_channel_stddev(self):
        if self.using_unlabeled_training_data:
            return np.array(self.classification_data_set_config['per_channel_stddev'], dtype=np.float32)
        else:
            return np.array(self.classification_data_set_config['per_channel_stddev_labeled'], dtype=np.float32)

    @property
    def class_weights(self):
        # Lazy load class weights when needed
        if self._class_weights is None:
            if self.use_class_weights:
                override_class_weights = self._get_config_value('override_class_weights')

                if override_class_weights is None:
                    raise ValueError('The ClassificationTrainer can only use override class weights and they were not found from the config file')

                self.logger.log('Using class weights: {}'.format(override_class_weights))
                class_weights = np.array(override_class_weights, dtype=np.float32)
                self._class_weights = class_weights
            else:
                self._class_weights = np.ones([self.num_classes], dtype=np.float32)

        return self._class_weights

    @property
    def ignore_classes(self):
        if self._ignore_classes is None:
            ignore_classes = np.squeeze(np.where(np.equal(self.class_weights, 0.0))).astype(dtype=np.int32)

            # If there is only a single ignored class the shape will be empty - expand to dim 1
            if ignore_classes.shape == ():
                ignore_classes = np.expand_dims(ignore_classes, axis=-1)

            self._ignore_classes = ignore_classes

        return self._ignore_classes

    @property
    def is_supervised_only_trainer(self):
        return self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED or \
               self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED_MEAN_TEACHER

    @property
    def using_unlabeled_training_data(self):
        return not self.is_supervised_only_trainer and \
               self.training_set_unlabeled is not None and \
               self.training_set_unlabeled.size > 0 and \
               self.num_unlabeled_per_batch > 0

    @property
    def training_data_generator_params(self):
        if self._training_data_generator_params is None:
            # Note: the initial epoch is an index but the property is a number starting from 1
            self._training_data_generator_params = DataGeneratorParameters(
                num_color_channels=self.num_color_channels,
                name='tr',
                random_seed=self.random_seed,
                crop_shapes=self.crop_shape,
                resize_shapes=self.resize_shape,
                use_per_channel_mean_normalization=True,
                per_channel_mean=self.per_channel_mean,
                use_per_channel_stddev_normalization=True,
                per_channel_stddev=self.per_channel_stddev,
                use_data_augmentation=self.use_data_augmentation,
                data_augmentation_params=self.data_augmentation_parameters,
                shuffle_data_after_epoch=True,
                div2_constraint=self.div2_constraint,
                initial_epoch=self.initial_epoch,
                generate_mean_teacher_data=self.using_mean_teacher_method,
                log_images_folder_path=self.logger.log_images_folder_path)

        return self._training_data_generator_params

    @property
    def validation_data_generator_params(self):
        if self._validation_data_generator_params is None:
            self._validation_data_generator_params = DataGeneratorParameters(
                num_color_channels=self.num_color_channels,
                name='val',
                random_seed=self.random_seed,
                crop_shapes=self.validation_crop_shape,
                resize_shapes=self.validation_resize_shape,
                use_per_channel_mean_normalization=True,
                per_channel_mean=self.per_channel_mean,
                use_per_channel_stddev_normalization=True,
                per_channel_stddev=self.per_channel_stddev,
                use_data_augmentation=False,
                data_augmentation_params=None,
                shuffle_data_after_epoch=True,
                div2_constraint=self.div2_constraint,
                generate_mean_teacher_data=False,
                log_images_folder_path=self.logger.log_images_folder_path)

        return self._validation_data_generator_params

    # TrainerBase implementations

    @property
    def num_unlabeled_per_batch(self):
        num_unlabeled_per_batch = int(self._get_config_value('num_unlabeled_per_batch'))

        # Sanity check for number of unlabeled per batch
        if num_unlabeled_per_batch > 0 and self.is_supervised_only_trainer:
            self.logger.warn('Trainer type is marked as {} with unlabeled per batch {} - assuming 0 unlabeled per batch'.format(self.trainer_type, num_unlabeled_per_batch))
            return 0

        return num_unlabeled_per_batch

    def _get_data_sets(self):
        # type: () -> (object, object, object, object)

        """
        Creates and initializes the data sets and return a tuple of four data sets:
        (training labeled, training unlabeled, validation and test)

        # Arguments
            None
        # Returns
            :return: A tuple of data sets (training labeled, training unlabeled, validation, test)
        """
        self.logger.log('Creating labeled training set')
        stime = time.time()
        training_set_labeled = MINCDataSet(name='training_set_labeled',
                                           path_to_photo_archive=self.path_to_labeled_photos,
                                           label_mappings_file_path=self.path_to_label_mapping_file,
                                           data_set_file_path=self.path_to_training_set_file)
        self.logger.log('Labeled training set construction took: {} s, size: {}'.format(time.time()-stime, training_set_labeled.size))

        if training_set_labeled.size == 0:
            raise ValueError('No training data found')

        # Unlabeled training set - skip construction if unlabeled per batch is zero or this is a supervised segmentation trainer
        if self.num_unlabeled_per_batch > 0 and not self.is_supervised_only_trainer:
            self.logger.log('Creating unlabeled training set from: {}'.format(self.path_to_unlabeled_photos))
            stime = time.time()
            training_set_unlabeled = UnlabeledImageDataSet(name='training_set_unlabeled',
                                                           path_to_photo_archive=self.path_to_unlabeled_photos)
            self.logger.log('Unlabeled training set creation took: {} s, size: {}'.format(time.time()-stime, training_set_unlabeled.size))
        else:
            training_set_unlabeled = None

        self.logger.log('Creating validation set')
        stime = time.time()
        validation_set = MINCDataSet(name='validation_set',
                                     path_to_photo_archive=self.path_to_labeled_photos,
                                     label_mappings_file_path=self.path_to_label_mapping_file,
                                     data_set_file_path=self.path_to_validation_set_file)
        self.logger.log('Validation set construction took: {} s, size: {}'.format(time.time()-stime, validation_set.size))

        #self.logger.log('Creating test set')
        #stime = time.time()
        #test_set = MINCDataSet(name='test_set',
        #                       path_to_photo_archive=self.path_to_labeled_photos,
        #                       label_mappings_file_path=self.path_to_label_mapping_file,
        #                       data_set_file_path=self.path_to_test_set_file)
        #self.logger.log('Test set construction took: {} s, size: {}'.format(time.time()-stime, test_set.size))

        return training_set_labeled, training_set_unlabeled, validation_set

    def _get_data_generators(self):
        # type: () -> (DataGenerator, DataGenerator)

        """
        Creates and initializes the data generators and returns a tuple of two data generators:
        (training data generator, validation data generator)

        # Arguments
            None
        # Returns
            :return: A tuple of two data generators (training, validation)
        """
        self.logger.log('Initializing data generators')

        self.logger.log('Creating training data generator')
        training_data_generator = ClassificationDataGenerator(labeled_data_set=self.training_set_labeled,
                                                              unlabeled_data_set=self.training_set_unlabeled if self.using_unlabeled_training_data else None,
                                                              num_labeled_per_batch=self.num_labeled_per_batch,
                                                              num_unlabeled_per_batch=self.num_unlabeled_per_batch if self.using_unlabeled_training_data else 0,
                                                              class_weights=self.class_weights,
                                                              batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
                                                              params=self.training_data_generator_params)

        self.logger.log('Creating validation data generator')
        validation_data_generator = ClassificationDataGenerator(labeled_data_set=self.validation_set,
                                                                unlabeled_data_set=None,
                                                                num_labeled_per_batch=self.validation_num_labeled_per_batch,
                                                                num_unlabeled_per_batch=0,
                                                                class_weights=self.class_weights,
                                                                batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
                                                                params=self.validation_data_generator_params)

        self.logger.log('Using unlabeled training data: {}'.format(self.using_unlabeled_training_data))
        self.logger.log('Using per-channel mean: {}'.format(training_data_generator.per_channel_mean))
        self.logger.log('Using per-channel stddev: {}'.format(training_data_generator.per_channel_stddev))

        return training_data_generator, validation_data_generator

    @property
    def num_classes(self):
        # type: () -> int
        return self.training_set_labeled.num_classes

    @property
    def total_batch_size(self):
        if self.using_unlabeled_training_data:
            return self.num_labeled_per_batch + self.num_unlabeled_per_batch

        return self.num_labeled_per_batch

    def train(self):
        # type: () -> History
        super(ClassificationTrainer, self).train()
        assert isinstance(self.model, ExtendedModel)

        # Labeled data set size determines the epochs
        num_workers = dataset_utils.get_number_of_parallel_jobs()-1

        if self.using_unlabeled_training_data:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}, unlabeled data set size: {}, num unlabeled per batch: {}'
                            .format(self.training_set_labeled.size, self.num_labeled_per_batch, self.training_set_unlabeled.size, self.num_unlabeled_per_batch))
        else:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}'.format(self.training_set_labeled.size, self.num_labeled_per_batch))

        self.logger.log('Num epochs: {}, initial epoch: {}, total batch size: {}, crop shape: {}, training steps per epoch: {}, validation steps per epoch: {}'
                        .format(self.num_epochs, self.initial_epoch, self.total_batch_size, self.crop_shape, self.training_steps_per_epoch, self.validation_steps_per_epoch))

        self.logger.log('Num workers: {}'.format(num_workers))

        # Get a list of callbacks
        callbacks = self._get_training_callbacks()

        # Note: the student model should not be evaluated using the validation data generator
        # the generator input will not be meaning
        history = self.model.fit_generator(
            generator=self.training_data_iterator,
            steps_per_epoch=self.training_steps_per_epoch,
            epochs=self.num_epochs,
            initial_epoch=self.initial_epoch,
            validation_data=self.validation_data_iterator,
            validation_steps=self.validation_steps_per_epoch,
            verbose=1,
            trainer=self,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=num_workers)

        return history

    def _get_model_lambda_loss_type(self):
        # type: () -> ModelLambdaLossType
        if self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED:
            return ModelLambdaLossType.CLASSIFICATION_CATEGORICAL_CROSS_ENTROPY
        elif self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED_MEAN_TEACHER:       # Same lambda loss as semi-supervised but only with 0 unlabeled data
            return ModelLambdaLossType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER
        elif self.trainer_type == TrainerType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER:
            return ModelLambdaLossType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER
        else:
            raise ValueError('Unsupported trainer type for ClassificationTrainer - cannot deduce lambda loss type')

    def _get_model_metrics(self):
        return {'logits': [metrics.classification_accuracy(self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                           metrics.classification_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes)]}

    def _get_model_loss(self):
        # type: () -> dict[str, Callable]
        return {'loss': losses.dummy_loss, 'logits': lambda _, y_pred: 0.0*y_pred}

    def _get_model_loss_weights(self):
        # type: () -> dict[str, float]
        return {'loss': 1., 'logits': 0.}

    # End of: TrainerBase implementations

    # MeanTeacherBase implementations

    def _get_teacher_validation_data_generator(self):
        if self.using_mean_teacher_method:
            teacher_validation_data_generator = ClassificationDataGenerator(labeled_data_set=self.validation_set,
                                                                            unlabeled_data_set=None,
                                                                            num_labeled_per_batch=self.validation_num_labeled_per_batch,
                                                                            num_unlabeled_per_batch=0,
                                                                            class_weights=self.class_weights,
                                                                            batch_data_format=BatchDataFormat.SUPERVISED,
                                                                            params=self.validation_data_generator_params)
        else:
            teacher_validation_data_generator = None

        return teacher_validation_data_generator

    @property
    def using_mean_teacher_method(self):
        return self.trainer_type == TrainerType.CLASSIFICATION_SUPERVISED_MEAN_TEACHER or \
               self.trainer_type == TrainerType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER

    def _get_teacher_model_lambda_loss_type(self):
        # type: () -> ModelLambdaLossType
        return ModelLambdaLossType.NONE

    def _get_teacher_model_loss(self):
        # type: () -> Callable
        return losses.classification_weighted_categorical_crossentropy_loss(self.class_weights)

    def _get_teacher_model_metrics(self):
        # type: () -> list[Callable]
        return [metrics.classification_accuracy(self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes),
                metrics.classification_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=self.ignore_classes)]

    # End of: MeanTeacherBase implementations

    def handle_early_exit(self):
        super(ClassificationTrainer, self).handle_early_exit()
        self.logger.log('Early exit handler complete - ready for exit')
        self.logger.close_log()

    def modify_batch_data(self, step_index, x, y, validation=False):
        # type: (int, list[np.array[np.float32]], np.array, bool) -> (list[np.array[np.float32]], np.array)

        """
        Invoked by the ExtendedModel right before train_on_batch:

        If using the Mean Teacher method:
            Modifies the batch data by appending the mean teacher predictions as the last
            element of the input data X if we are using mean teacher training.

        # Arguments
            :param step_index: the training step index
            :param x: input data
            :param y: output data
            :param validation: is this a validation data batch
        # Returns
            :return: a tuple of (input data, output data)
        """
        x, y = super(ClassificationTrainer, self).modify_batch_data(step_index=step_index, x=x, y=y, validation=validation)
        return x, y
