# coding = utf-8

import os
import json
import random
import datetime
import time
import numpy as np

from enum import Enum
from PIL import ImageFile
from abc import ABCMeta, abstractmethod

import keras
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping

from tensorflow.python.client import timeline

from callbacks.optimizer_checkpoint import OptimizerCheckpoint
from callbacks.stepwise_learning_rate_scheduler import StepwiseLearningRateScheduler
from models.extended_model import ExtendedModel
from generators import SegmentationDataGenerator, MINCClassificationDataGenerator
from generators import DataGeneratorParameters, DataAugmentationParameters, BatchDataFormat

from utils import dataset_utils
from models.models import ModelLambdaLossType, get_model
from utils import image_utils
from utils import general_utils

from data_set import LabeledImageDataSet, UnlabeledImageDataSet
from utils.dataset_utils import MaterialClassInformation, SegmentationDataSetInformation
import losses
import metrics
import settings
from logger import Logger, LogLevel

#############################################
# TIME LINER
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
    CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER = 6


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
        self.trainer_type = TrainerType[trainer_type.upper()]
        self.model_name = model_name
        self.model_folder_name = model_folder_name
        self.last_completed_epoch = -1

        # Profiling related variables
        self.profiling_timeliner = TimeLiner() if settings.PROFILE else None
        self.profiling_run_metadata = K.tf.RunMetadata() if settings.PROFILE else None
        self.profiling_run_options = K.tf.RunOptions(trace_level=K.tf.RunOptions.FULL_TRACE) if settings.PROFILE else None

        # Without this some truncated images can throw errors
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.config = self._load_config_json(config_file_path)

        # Setup the log file path to enable logging
        log_file_path = self.get_config_value('log_file_path').format(model_folder=self.model_folder_name)
        log_to_stdout = self.get_config_value('log_to_stdout')
        self.logger = Logger(log_file_path=log_file_path, use_timestamp=True, log_to_stdout_default=log_to_stdout)

        # Log the Keras and Tensorflow versions
        self.logger.log('\n\n############################################################\n\n')
        self.logger.log('Using Keras version: {}'.format(keras.__version__))
        self.logger.log('Using Tensorflow version: {}'.format(K.tf.__version__))

        # Seed the random in order to be able to reproduce the results
        # Note: both random and np.random
        self.random_seed = int(self.get_config_value('random_seed'))
        self.logger.log('Initializing random and np.random with random seed: {}'.format(self.random_seed))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Set image data format
        self.logger.log('Setting Keras image data format to: {}'.format(self.get_config_value('image_data_format')))
        K.set_image_data_format(self.get_config_value('image_data_format'))

        # Parse data augmentation parameters
        if self.get_config_value('use_data_augmentation'):
            self.logger.log('Parsing data augmentation parameters')

            augmentation_config = self.get_config_value('data_augmentation_params')

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

            self.data_augmentation_parameters = DataAugmentationParameters(
                augmentation_probability_function=augmentation_probability_function,
                rotation_range=rotation_range,
                zoom_range=zoom_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                channel_shift_range=channel_shift_range,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                gaussian_noise_stddev_function=gaussian_noise_stddev_function,
                gamma_adjust_range=gamma_adjust_range)

            self.logger.log('Data augmentation params: augmentation probability function: {}, rotation range: {}, zoom range: {}, '
                            'width shift range: {}, height shift range: {}, channel shift range: {}, horizontal flip: {}, vertical flip: {},'
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
        else:
            self.data_augmentation_parameters = None

        self._init_config()
        self._init_data()
        self._init_models()
        self._init_data_generators()

    @abstractmethod
    def _init_config(self):
        self.logger.log('Reading configuration file')
        self.save_values_on_early_exit = self.get_config_value('save_values_on_early_exit')

    @abstractmethod
    def _init_data(self):
        self.logger.log('Initializing data')

    @abstractmethod
    def _init_models(self):
        self.logger.log('Initializing models')

    @abstractmethod
    def _init_data_generators(self):
        self.logger.log('Initializing data generators')

    @staticmethod
    def _load_config_json(path):
        with open(path) as f:
            data = f.read()
            return json.loads(data)

    def get_config_value(self, key):
        return self.config[key] if key in self.config else None

    def set_config_value(self, key, value):
        self.config[key] = value

    def get_callbacks(self):
        keras_model_checkpoint = self.get_config_value('keras_model_checkpoint')
        keras_tensorboard_log_path = self.get_config_value('keras_tensorboard_log_path').format(model_folder=self.model_folder_name)
        keras_csv_log_file_path = self.get_config_value('keras_csv_log_file_path').format(model_folder=self.model_folder_name)
        early_stopping = self.get_config_value('early_stopping')
        reduce_lr_on_plateau = self.get_config_value('reduce_lr_on_plateau')
        stepwise_learning_rate_scheduler = self.get_config_value('stepwise_learning_rate_scheduler')
        optimizer_checkpoint_file_path = self.get_config_value('optimizer_checkpoint_file_path').format(model_folder=self.model_folder_name)

        callbacks = []

        # Always ensure that the model checkpoint has been provided
        if not keras_model_checkpoint:
            raise ValueError('Could not find Keras ModelCheckpoint configuration with key keras_model_checkpoint - would be unable to save model weights')

        # Make sure the model checkpoints directory exists
        keras_model_checkpoint_dir = os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')).format(model_folder=self.model_folder_name)
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
            general_utils.create_path_if_not_existing(keras_tensorboard_log_path)

            tensorboard_checkpoint_callback = TensorBoard(
                log_dir=keras_tensorboard_log_path,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_grads=False,  # Note: writing grads for a bit network takes about an hour
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None)

            callbacks.append(tensorboard_checkpoint_callback)

        # CSV logger for streaming epoch results
        if keras_csv_log_file_path is not None:
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
                                                                  verbose=verbose)

            callbacks.append(stepwise_lr_scheduler)

        # Optimizer checkpoint
        if optimizer_checkpoint_file_path is not None:
            general_utils.create_path_if_not_existing(optimizer_checkpoint_file_path)
            optimizer_checkpoint = OptimizerCheckpoint(optimizer_checkpoint_file_path)
            callbacks.append(optimizer_checkpoint)

        return callbacks

    @property
    def model_checkpoint_directory(self):
        keras_model_checkpoint = self.get_config_value('keras_model_checkpoint')
        keras_model_checkpoint_dir = os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')).format(model_folder=self.model_folder_name)
        return keras_model_checkpoint_dir

    @property
    def model_checkpoint_file_path(self):
        keras_model_checkpoint = self.get_config_value('keras_model_checkpoint')
        keras_model_checkpoint_dir = os.path.dirname(keras_model_checkpoint.get('checkpoint_file_path')).format(model_folder=self.model_folder_name)
        keras_model_checkpoint_file = os.path.basename(keras_model_checkpoint.get('checkpoint_file_path'))
        keras_model_checkpoint_file_path = os.path.join(keras_model_checkpoint_dir, keras_model_checkpoint_file)
        return keras_model_checkpoint_file_path

    def get_optimizer(self, continue_from_optimizer_checkpoint):
        optimizer_info = self.get_config_value('optimizer')
        optimizer_configuration = None
        optimizer = None
        optimizer_name = optimizer_info['name'].lower()

        if continue_from_optimizer_checkpoint:
            optimizer_configuration_file_path = self.get_config_value('optimizer_checkpoint_file_path')
            self.logger.log('Loading optimizer configuration from file: {}'.format(optimizer_configuration_file_path))

            try:
                with open(optimizer_configuration_file_path, 'r') as f:
                    data = f.read()
                    optimizer_configuration = json.loads(data)
            except IOError as e:
                self.logger.log('Could not load optimizer configuration from file: {}, error: {}. Continuing without config.'
                         .format(optimizer_configuration_file_path, e.message))
                optimizer_configuration = None

        if optimizer_name == 'adam':
            if optimizer_configuration is not None:
                optimizer = Adam.from_config(optimizer_configuration)
            else:
                lr = optimizer_info['learning_rate']
                decay = optimizer_info['decay']
                optimizer = Adam(lr=lr, decay=decay)

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
                optimizer = SGD(lr=lr, momentum=momentum, decay=decay)

            self.logger.log('Using {} optimizer with learning rate: {}, momentum: {}, decay: {}'
                .format(optimizer.__class__.__name__,
                        K.get_value(optimizer.lr),
                        K.get_value(optimizer.momentum),
                        K.get_value(optimizer.decay)))

        else:
            raise ValueError('Unsupported optimizer name: {}'.format(optimizer_name))

        return optimizer

    @staticmethod
    def get_latest_weights_file_path(weights_folder_path):
        weight_files = dataset_utils.get_files(weights_folder_path)

        if len(weight_files) > 0:
            weight_files.sort()
            weight_file = weight_files[-1]

            if os.path.isfile(os.path.join(weights_folder_path, weight_file)) and (".hdf5" in weight_file):
                return os.path.join(weights_folder_path, weight_file)

        return None

    def load_latest_weights_for_model(self, model, weights_directory_path):
        initial_epoch = 0

        try:
            # Try to find weights from the checkpoint path
            if os.path.isdir(weights_directory_path):
                weights_folder = weights_directory_path
            else:
                weights_folder = os.path.dirname(weights_directory_path)

            self.logger.log('Searching for existing weights from checkpoint path: {}'.format(weights_folder))
            weight_file_path = TrainerBase.get_latest_weights_file_path(weights_folder)

            if weight_file_path is None:
                self.logger.log('Could not locate any suitable weight files from the given path')
                return 0

            weight_file = weight_file_path.split('/')[-1]

            if weight_file:
                self.logger.log('Loading network weights from file: {}'.format(weight_file_path))
                model.load_weights(weight_file_path)

                # Parse the epoch number: <epoch>-<val_loss>
                epoch_val_loss = weight_file.split('.')[1]
                initial_epoch = int(epoch_val_loss.split('-')[0]) + 1
                self.logger.log('Continuing training from epoch: {}'.format(initial_epoch))
            else:
                self.logger.log('No existing weights were found')

        except Exception as e:
            self.logger.log('Searching for existing weights finished with an error: {}'.format(e.message))
            return 0

        return initial_epoch

    def transfer_weights(self, to_model_wrapper, transfer_weights_options):
        # type: (ModelBase, dict) -> ()

        transfer_model_name = transfer_weights_options['transfer_model_name']
        transfer_model_input_shape = tuple(transfer_weights_options['transfer_model_input_shape'])
        transfer_model_num_classes = transfer_weights_options['transfer_model_num_classes']
        transfer_model_weights_file_path = transfer_weights_options['transfer_model_weights_file_path']

        self.logger.log('Creating transfer model: {} with input shape: {}, num classes: {}'
                        .format(transfer_model_name, transfer_model_input_shape, transfer_model_num_classes))
        transfer_model_wrapper = get_model(transfer_model_name,
                                           transfer_model_input_shape,
                                           transfer_model_num_classes)
        transfer_model = transfer_model_wrapper.model
        transfer_model.summary()

        self.logger.log('Loading transfer weights to transfer model from file: {}'.format(transfer_model_weights_file_path))
        transfer_model.load_weights(transfer_model_weights_file_path)

        from_layer_index = transfer_weights_options['from_layer_index']
        to_layer_index = transfer_weights_options['to_layer_index']
        freeze_transferred_layers = transfer_weights_options['freeze_transferred_layers']
        self.logger.log('Transferring weights from layer range: [{}:{}], freeze transferred layers: {}'
            .format(from_layer_index, to_layer_index, freeze_transferred_layers))

        transferred_layers, last_transferred_layer = to_model_wrapper._transfer_weights(
            from_model=transfer_model,
            from_layer_index=from_layer_index,
            to_layer_index=to_layer_index,
            freeze_transferred_layers=freeze_transferred_layers)

        self.logger.log('Weight transfer completed with {} transferred layers, last transferred layer: {}'
            .format(transferred_layers, last_transferred_layer))

    @abstractmethod
    def train(self):
        self.logger.log('Starting training at local time {}\n'.format(datetime.datetime.now()))

        if settings.DEBUG:
            self.logger.debug_log('Training in debug mode')

        if settings.PROFILE:
            self.logger.profile_log('Training in profiling mode')

            graph_def_file_folder = self.logger.log_folder_path
            self.logger.profile_log('Writing Tensorflow GraphDef to: {}'.format(os.path.join(graph_def_file_folder, "graph_def")))
            K.tf.train.write_graph(K.get_session().graph_def, graph_def_file_folder, "graph_def", as_text=True)
            self.logger.profile_log('Writing Tensorflow GraphDef complete')

    def handle_early_exit(self):
        self.logger.log('Handle early exit method called')

        if not self.save_values_on_early_exit:
            self.logger.log('Save values on early exit is disabled')
            return

    def get_compile_kwargs(self):
        if settings.PROFILE:
            return {'options': self.profiling_run_options, 'run_metadata': self.profiling_run_metadata}
        return {}

    def modify_batch_data(self, step_index, x, y, validation=False):
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

    def save_optimizer_settings(self, model, file_extension=''):
        file_path = self.get_config_value('optimizer_checkpoint_file_path').format(model_folder=self.model_folder_name) + file_extension

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
            profiling_timeline_file_path = os.path.join(self.logger.log_folder_path, 'profiling_timeline.json')
            self.logger.profile_log('Saving profiling data to: {}'.format(profiling_timeline_file_path))
            self.profiling_timeliner.save(profiling_timeline_file_path)

#############################################
# SEGMENTATION TRAINER
#############################################


class SegmentationTrainer(TrainerBase):

    def __init__(self,
                 trainer_type,
                 model_name,
                 model_folder_name,
                 config_file_path):
        # type: (str, str, str, str, str) -> ()

        # Declare variables that are going to be initialized in the _init_ functions
        self.label_generation_function = None
        self.consistency_cost_coefficient_function = None
        self.ema_smoothing_coefficient_function = None
        self.unlabeled_cost_coefficient_function = None

        self.material_class_information = None
        self.data_set_information = None
        self.num_classes = -1

        self.labeled_photo_files = None
        self.labeled_mask_files = None
        self.unlabeled_photo_files = None

        self.training_set_labeled = None
        self.training_set_unlabeled = None
        self.validation_set = None
        self.test_set = None

        self.model_wrapper = None
        self.model = None
        self.teacher_model_wrapper = None
        self.teacher_model = None
        self.initial_epoch = 0

        self.training_data_generator = None
        self.validation_data_generator = None
        self.teacher_validation_data_generator = None

        super(SegmentationTrainer, self).__init__(trainer_type, model_name, model_folder_name, config_file_path)

    def _init_config(self):
        super(SegmentationTrainer, self)._init_config()

        self.path_to_material_class_file = self.get_config_value('path_to_material_class_file')
        self.path_to_data_set_information_file = self.get_config_value('path_to_data_set_information_file')
        self.path_to_labeled_photos = self.get_config_value('path_to_labeled_photos')
        self.path_to_labeled_masks = self.get_config_value('path_to_labeled_masks')
        self.path_to_unlabeled_photos = self.get_config_value('path_to_unlabeled_photos')
        self.use_class_weights = bool(self.get_config_value('use_class_weights'))
        self.use_material_samples = bool(self.get_config_value('use_material_samples'))

        self.input_shape = self.get_config_value('input_shape')
        self.continue_from_last_checkpoint = bool(self.get_config_value('continue_from_last_checkpoint'))
        self.use_transfer_weights = bool(self.get_config_value('transfer_weights'))
        self.transfer_options = self.get_config_value('transfer_options')
        self.continue_from_optimizer_checkpoint = bool(self.get_config_value('continue_from_optimizer_checkpoint'))
        self.loss_function_name = self.get_config_value('loss_function')

        # If using mean teacher method read the parameters - any missing parameters should raise
        # exceptions
        if self.using_mean_teacher_method:
            self._init_mean_teacher_config()

        # If using superpixel method; read the parameters - any missing parameters should raise
        # exceptions
        if self.using_superpixel_method:
            self._init_superpixel_config()

        self.use_data_augmentation = bool(self.get_config_value('use_data_augmentation'))
        self.num_color_channels = self.get_config_value('num_color_channels')

        self.num_epochs = self.get_config_value('num_epochs')
        self.num_labeled_per_batch = self.get_config_value('num_labeled_per_batch')
        self.num_unlabeled_per_batch = self.get_config_value('num_unlabeled_per_batch')
        self.crop_shape = self.get_config_value('crop_shape')
        self.resize_shape = self.get_config_value('resize_shape')
        self.validation_num_labeled_per_batch = self.get_config_value('validation_num_labeled_per_batch')
        self.validation_crop_shape = self.get_config_value('validation_crop_shape')
        self.validation_resize_shape = self.get_config_value('validation_resize_shape')

        # Sanity check for number of unlabeled per batch
        if self.num_unlabeled_per_batch > 0 and self.is_supervised_only_trainer:
            self.logger.warn('Trainer type is marked as {} with unlabeled per batch {} - assuming 0 unlabeled per batch'.format(self.trainer_type, self.num_unlabeled_per_batch))
            self.num_unlabeled_per_batch = 0

    def _init_mean_teacher_config(self):
        self.logger.log('Reading mean teacher method configuration')
        mean_teacher_params = self.get_config_value('mean_teacher_params')

        if mean_teacher_params is None:
            raise ValueError('Could not find entry for mean_teacher_params from the configuration JSON')

        teacher_weights_directory_path = os.path.dirname(mean_teacher_params['teacher_model_checkpoint_file_path']).format(model_folder=self.model_folder_name)
        self.logger.log('Teacher weights directory path: {}'.format(teacher_weights_directory_path))
        self.teacher_weights_directory_path = teacher_weights_directory_path
        self.teacher_model_checkpoint_file_path = mean_teacher_params['teacher_model_checkpoint_file_path']

        ema_coefficient_schedule_function = mean_teacher_params['ema_smoothing_coefficient_function']
        self.logger.log('EMA smoothing coefficient function: {}'.format(ema_coefficient_schedule_function))
        self.ema_smoothing_coefficient_function = eval(ema_coefficient_schedule_function)

        consistency_cost_coefficient_function = mean_teacher_params['consistency_cost_coefficient_function']
        self.logger.log('Consistency cost coefficient function: {}'.format(consistency_cost_coefficient_function))
        self.consistency_cost_coefficient_function = eval(consistency_cost_coefficient_function)

    def _init_superpixel_config(self):
        self.logger.log('Reading superpixel method configuration')
        superpixel_params = self.get_config_value('superpixel_params')

        if superpixel_params is None:
            raise ValueError('Could not find entry for superpixel_params from the configuration JSON')

        label_generation_function_name = superpixel_params['label_generation_function_name']
        self.logger.log('Label generation function name: {}'.format(label_generation_function_name))
        self.label_generation_function = self.get_label_generation_function(label_generation_function_name)

        unlabeled_cost_coefficient_function = superpixel_params['unlabeled_cost_coefficient_function']
        self.logger.log('Unlabeled cost coefficient function: {}'.format(unlabeled_cost_coefficient_function))
        self.unlabeled_cost_coefficient_function = eval(unlabeled_cost_coefficient_function)

    def _init_data(self):
        super(SegmentationTrainer, self)._init_data()

        # Load material class information
        self.logger.log('Loading material class information from: {}'.format(self.path_to_material_class_file))
        self.material_class_information = dataset_utils.load_material_class_information(self.path_to_material_class_file)
        self.num_classes = len(self.material_class_information)
        self.logger.log('Loaded {} material classes successfully'.format(self.num_classes))

        # Load data set information
        self.logger.log('Loading data set information from: {}'.format(self.path_to_data_set_information_file))
        self.data_set_information = dataset_utils.load_segmentation_data_set_information(self.path_to_data_set_information_file)
        self.logger.log('Loaded data set information successfully with set sizes (tr,va,te): {}, {}, {}'
                 .format(self.data_set_information.training_set.labeled_size + self.data_set_information.training_set.unlabeled_size,
                         self.data_set_information.validation_set.labeled_size,
                         self.data_set_information.test_set.labeled_size))

        self.logger.log('Constructing labeled data sets with photo files from: {} and mask files from: {}'.format(self.path_to_labeled_photos, self.path_to_labeled_masks))

        # Labeled training set
        self.logger.log('Constructing labeled training set')
        stime = time.time()
        self.training_set_labeled = LabeledImageDataSet('training_set_labeled',
                                                        path_to_photo_archive=self.path_to_labeled_photos,
                                                        path_to_mask_archive=self.path_to_labeled_masks,
                                                        photo_file_list=self.data_set_information.training_set.labeled_photos,
                                                        mask_file_list=self.data_set_information.training_set.labeled_masks,
                                                        material_samples=self.data_set_information.training_set.material_samples)
        self.logger.log('Labeled training set construction took: {} s, size: {}'.format(time.time()-stime, self.training_set_labeled.size))

        if self.training_set_labeled.size == 0:
            raise ValueError('No training data found')

        # Unlabeled training set - skip construction if unlabeled per batch is zero or this is a supervised segmentation trainer
        if self.num_unlabeled_per_batch > 0 and self.trainer_type != TrainerType.SEGMENTATION_SUPERVISED:
            self.logger.log('Constructing unlabeled training set from: {}'.format(self.path_to_unlabeled_photos))
            stime = time.time()
            self.training_set_unlabeled = UnlabeledImageDataSet('training_set_unlabeled',
                                                                path_to_photo_archive=self.path_to_unlabeled_photos,
                                                                photo_file_list=self.data_set_information.training_set.unlabeled_photos)
            self.logger.log('Unlabeled training set construction took: {} s, size: {}'.format(time.time()-stime, self.training_set_unlabeled.size))
        else:
            self.training_set_unlabeled = None

        # Labeled validation set
        self.logger.log('Constructing validation set')
        stime = time.time()
        self.validation_set = LabeledImageDataSet('validation_set',
                                                  self.path_to_labeled_photos,
                                                  self.path_to_labeled_masks,
                                                  photo_file_list=self.data_set_information.validation_set.labeled_photos,
                                                  mask_file_list=self.data_set_information.validation_set.labeled_masks,
                                                  material_samples=self.data_set_information.validation_set.material_samples)
        self.logger.log('Labeled validation set construction took: {} s, size: {}'.format(time.time()-stime, self.validation_set.size))

        # Labeled test set
        self.logger.log('Constructing test set')
        stime = time.time()
        self.test_set = LabeledImageDataSet('test_set',
                                            self.path_to_labeled_photos,
                                            self.path_to_labeled_masks,
                                            photo_file_list=self.data_set_information.test_set.labeled_photos,
                                            mask_file_list=self.data_set_information.test_set.labeled_masks,
                                            material_samples=self.data_set_information.test_set.material_samples)
        self.logger.log('Labeled test set construction took: {} s, size: {}'.format(time.time()-stime, self.test_set.size))

        if self.training_set_unlabeled is not None:
            total_data_set_size = self.training_set_labeled.size + self.training_set_unlabeled.size + self.validation_set.size + self.test_set.size
        else:
            total_data_set_size = self.training_set_labeled.size + self.validation_set.size + self.test_set.size

        self.logger.log('Total data set size: {}'.format(total_data_set_size))

        # Class weights
        self.class_weights = self.get_class_weights(data_set_information=self.data_set_information)

    def _init_models(self):
        super(SegmentationTrainer, self)._init_models()

        # Model creation
        student_model_type = self.get_model_lambda_loss_type()

        self.logger.log('Creating student model {} instance with type: {}, input shape: {}, num classes: {}'
                        .format(self.model_name, student_model_type, self.input_shape, self.num_classes))
        self.logger.log('Using mean teacher method: {}'.format(self.using_mean_teacher_method))
        self.logger.log('Using superpixel method: {}'.format(self.using_superpixel_method))

        self.model_wrapper = get_model(self.model_name,
                                       self.input_shape,
                                       self.num_classes,
                                       logger=self.logger,
                                       model_lambda_loss_type=student_model_type)

        self.model = self.model_wrapper.model
        self.model.summary()

        if self.continue_from_last_checkpoint:
            self.initial_epoch = self.load_latest_weights_for_model(self.model, self.model_checkpoint_directory)

        if self.use_transfer_weights:
            if self.initial_epoch != 0:
                self.logger.warn('Cannot transfer weights when continuing from last checkpoint. Skipping weight transfer')
            else:
                self.transfer_weights(self.model_wrapper, self.transfer_options)

        # Get the optimizer for the model
        if self.continue_from_optimizer_checkpoint and self.initial_epoch == 0:
            self.logger.warn('Cannot continue from optimizer checkpoint if initial epoch is 0. Ignoring optimizer checkpoint.')
            self.continue_from_optimizer_checkpoint = False

        optimizer = self.get_optimizer(self.continue_from_optimizer_checkpoint)

        # Get the loss function for the student model
        if self.loss_function_name != 'dummy':
            self.logger.warn('Semisupervised trainer should uses \'dummy\' loss function, got: {}. Ignoring passed loss function.'.format(self.loss_function_name))
            self.loss_function_name = 'dummy'

        loss_function = losses.dummy_loss

        # Ignore all the classes which have zero weights in the metrics
        ignore_classes = np.squeeze(np.where(np.equal(self.class_weights, 0.0))).astype(dtype=np.int32)

        if len(ignore_classes) > 0:
            self.logger.log('Ignoring classes: {}'.format(list(ignore_classes)))

        # Compile the student model
        self.model.compile(optimizer=optimizer,
                           loss={'loss': loss_function, 'logits': lambda _, y_pred: 0.0*y_pred},
                           loss_weights={'loss': 1., 'logits': 0.},
                           metrics={'logits': [metrics.segmentation_accuracy(self.num_unlabeled_per_batch, ignore_classes=ignore_classes),
                                               metrics.segmentation_mean_iou(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=ignore_classes),
                                               metrics.segmentation_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=ignore_classes)]},
                           **self.get_compile_kwargs())

        # If we are using the mean teacher method create the teacher model
        if self.using_mean_teacher_method:
            teacher_model_lambda_loss_type = ModelLambdaLossType.NONE
            self.logger.log('Creating teacher model {} instance with lambda loss type: {}, input shape: {}, num classes: {}'.format(self.model_name, teacher_model_lambda_loss_type, self.input_shape, self.num_classes))
            self.teacher_model_wrapper = get_model(self.model_name, self.input_shape, self.num_classes, logger=self.logger, model_lambda_loss_type=teacher_model_lambda_loss_type)
            self.teacher_model = self.teacher_model_wrapper.model
            self.teacher_model.summary()

            if self.continue_from_last_checkpoint:
                self.logger.log('Loading latest teacher model weights from path: {}'.format(self.teacher_weights_directory_path))
                initial_teacher_epoch = self.load_latest_weights_for_model(self.teacher_model, self.teacher_weights_directory_path)

                if initial_teacher_epoch < 1:
                    self.logger.warn('Could not find suitable weights, initializing teacher with student model weights')
                    self.teacher_model.set_weights(self.model.get_weights())
            else:
                self.teacher_model.set_weights(self.model.get_weights())

            # Note: Teacher model can use the regular metrics
            self.teacher_model.compile(optimizer=optimizer,
                                       loss=losses.segmentation_sparse_weighted_categorical_cross_entropy(self.class_weights),
                                       metrics=[metrics.segmentation_accuracy(self.num_unlabeled_per_batch, ignore_classes=ignore_classes),
                                                metrics.segmentation_mean_iou(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=ignore_classes),
                                                metrics.segmentation_mean_per_class_accuracy(self.num_classes, self.num_unlabeled_per_batch, ignore_classes=ignore_classes)],
                                       **self.get_compile_kwargs())

    def _init_data_generators(self):
        super(SegmentationTrainer, self)._init_data_generators()

        # Create training data and validation data generators
        # Note: training data comes from semi-supervised segmentation data generator and validation
        # and test data come from regular segmentation data generator
        self.logger.log('Creating training data generator')

        training_data_generator_params = DataGeneratorParameters(
            material_class_information=self.material_class_information,
            num_color_channels=self.num_color_channels,
            logger=self.logger,
            random_seed=self.random_seed,
            crop_shape=self.crop_shape,
            resize_shape=self.resize_shape,
            use_per_channel_mean_normalization=True,
            per_channel_mean=self.data_set_information.per_channel_mean if self.using_unlabeled_training_data else self.data_set_information.labeled_per_channel_mean,
            use_per_channel_stddev_normalization=True,
            per_channel_stddev=self.data_set_information.labeled_per_channel_stddev if self.using_unlabeled_training_data else self.data_set_information.labeled_per_channel_stddev,
            use_data_augmentation=self.use_data_augmentation,
            use_material_samples=self.use_material_samples,
            data_augmentation_params=self.data_augmentation_parameters,
            shuffle_data_after_epoch=True,
            div2_constraint=4)

        self.training_data_generator = SegmentationDataGenerator(
            labeled_data_set=self.training_set_labeled,
            unlabeled_data_set=self.training_set_unlabeled if self.using_unlabeled_training_data else None,
            num_labeled_per_batch=self.num_labeled_per_batch,
            num_unlabeled_per_batch=self.num_unlabeled_per_batch if self.using_unlabeled_training_data else 0,
            params=training_data_generator_params,
            class_weights=self.class_weights,
            batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
            label_generation_function=self.label_generation_function)

        self.logger.log('Creating validation data generator')

        validation_data_generator_params = DataGeneratorParameters(
            material_class_information=self.material_class_information,
            num_color_channels=self.num_color_channels,
            logger=self.logger,
            random_seed=self.random_seed,
            crop_shape=self.validation_crop_shape,
            resize_shape=self.validation_resize_shape,
            use_per_channel_mean_normalization=True,
            per_channel_mean=training_data_generator_params.per_channel_mean,
            use_per_channel_stddev_normalization=True,
            per_channel_stddev=training_data_generator_params.per_channel_stddev,
            use_data_augmentation=False,
            use_material_samples=False,
            data_augmentation_params=None,
            shuffle_data_after_epoch=True,
            div2_constraint=4)

        # The student lambda loss layer needs semi-supervised input, so we need to work around it
        # to only provide labeled input from the semi-supervised data generator. The dummy data
        # is appended to each batch so that the batch data maintains it's shape. This is done in the
        # modify_batch_data function.
        self.validation_data_generator = SegmentationDataGenerator(
            labeled_data_set=self.validation_set,
            unlabeled_data_set=None,
            num_labeled_per_batch=self.validation_num_labeled_per_batch,
            num_unlabeled_per_batch=0,
            params=validation_data_generator_params,
            class_weights=self.class_weights,
            batch_data_format=BatchDataFormat.SEMI_SUPERVISED,
            label_generation_function=None)

        # Note: The teacher has a supervised batch data format for validation data generation
        # because it doesn't have the semi-supervised loss lambda layer since we need to predict with it
        if self.using_mean_teacher_method:
            self.teacher_validation_data_generator = SegmentationDataGenerator(
                labeled_data_set=self.validation_set,
                unlabeled_data_set=None,
                num_labeled_per_batch=self.validation_num_labeled_per_batch,
                num_unlabeled_per_batch=0,
                params=validation_data_generator_params,
                class_weights=self.class_weights,
                batch_data_format=BatchDataFormat.SUPERVISED,
                label_generation_function=None)
        else:
            self.teacher_validation_data_generator = None

        self.logger.log('Using unlabeled training data: {}'.format(self.using_unlabeled_training_data))
        self.logger.log('Using material samples: {}'.format(self.training_data_generator.use_material_samples))
        self.logger.log('Using per-channel mean: {}'.format(self.training_data_generator.per_channel_mean))
        self.logger.log('Using per-channel stddev: {}'.format(self.training_data_generator.per_channel_stddev))

    def get_class_weights(self, data_set_information):
        # type: (SegmentationDataSetInformation) -> np.ndarray[np.float32]

        # Calculate class weights for the data if necessary
        if self.use_class_weights:

            class_weights = data_set_information.class_weights
            override_class_weights = self.get_config_value('override_class_weights')

            # Legacy support for data sets without material_samples_class_weights
            if self.use_material_samples and hasattr(data_set_information, 'material_samples_class_weights'):
                if data_set_information.material_samples_class_weights is not None and len(data_set_information.material_samples_class_weights) > 0:
                    class_weights = data_set_information.material_samples_class_weights
                else:
                    self.logger.warn('The trainer is using material samples but no material_samples_class_weights '
                                     'could be found - use override class weights or recreate the dataset.')
            elif self.use_material_samples and not hasattr(data_set_information, 'material_samples_class_weights'):
                self.logger.warn('The trainer is using material samples but no material_samples_class_weights '
                                 'could be found - use override class weights or recreate the dataset.')

            if override_class_weights is not None:
                self.logger.log('Found override class weights: {}'.format(override_class_weights))
                self.logger.log('Using override class weights instead of data set information class weights')
                class_weights = override_class_weights

            if class_weights is None or (len(class_weights) != self.num_classes):
                raise ValueError('Existing class weights were not found')
            if len(class_weights) != self.num_classes:
                raise ValueError('Number of classes in class weights did not match number of material classes: {} vs {}'
                                 .format(len(class_weights), self.num_classes))

            self.logger.log('Using class weights: {}'.format(class_weights))
            class_weights = np.array(class_weights, dtype=np.float32)
            return class_weights
        else:
            return np.ones([self.num_classes], dtype=np.float32)

    @property
    def is_supervised_only_trainer(self):
        return self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED or \
               self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED_MEAN_TEACHER

    @property
    def using_mean_teacher_method(self):
        return self.trainer_type == TrainerType.SEGMENTATION_SUPERVISED_MEAN_TEACHER or \
               self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER or \
               self.trainer_type == TrainerType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL

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
    def using_gaussian_noise(self):
        return self.data_augmentation_parameters is not None and self.data_augmentation_parameters.gaussian_noise_stddev_function is not None

    @property
    def total_batch_size(self):
        if not self.using_unlabeled_training_data:
            return self.num_labeled_per_batch

        return self.num_labeled_per_batch + self.num_unlabeled_per_batch

    def train(self):
        # type: () -> History
        super(SegmentationTrainer, self).train()

        assert isinstance(self.model, ExtendedModel)

        # Labeled data set size determines the epochs
        training_steps_per_epoch = self.training_data_generator.num_steps_per_epoch
        validation_steps_per_epoch = self.validation_data_generator.num_steps_per_epoch
        num_workers = dataset_utils.get_number_of_parallel_jobs()

        if self.using_unlabeled_training_data:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}, unlabeled data set size: {}, num unlabeled per batch: {}'
                     .format(self.training_set_labeled.size, self.num_labeled_per_batch, self.training_set_unlabeled.size, self.num_unlabeled_per_batch))
        else:
            self.logger.log('Labeled data set size: {}, num labeled per batch: {}'
                     .format(self.training_set_labeled.size, self.num_labeled_per_batch))

        self.logger.log('Num epochs: {}, initial epoch: {}, total batch size: {}, crop shape: {}, training steps per epoch: {}, validation steps per epoch: {}'
                 .format(self.num_epochs, self.initial_epoch, self.total_batch_size, self.crop_shape, training_steps_per_epoch, validation_steps_per_epoch))

        self.logger.log('Num workers: {}'.format(num_workers))

        # Get a list of callbacks
        callbacks = self.get_callbacks()

        # Note: the student model should not be evaluated using the validation data generator
        # the generator input will not be meaning
        history = self.model.fit_generator(
            generator=self.training_data_generator,
            steps_per_epoch=training_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH,
            epochs=self.num_epochs if not settings.PROFILE else settings.PROFILE_NUM_EPOCHS,
            initial_epoch=self.initial_epoch,
            validation_data=self.validation_data_generator,
            validation_steps=validation_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH,
            verbose=1,
            trainer=self,
            callbacks=callbacks,
            workers=num_workers)

        return history

    def handle_early_exit(self):
        super(SegmentationTrainer, self).handle_early_exit()

        if not self.save_values_on_early_exit:
            return

        # Stop training
        self.logger.log('Stopping model training')
        self.model.stop_fit_generator()

        # Save student model weights
        self.logger.log('Saving model weights')
        self.save_student_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.student-early-stop')

        # Save teacher model weights
        if self.using_mean_teacher_method:
            self.logger.log('Saving teacher model weights')
            self.save_teacher_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.teacher-early-stop')

        # Save optimizer settings
        self.logger.log('Saving model optimizer settings')
        self.save_optimizer_settings(model=self.model, file_extension='.student-early-stop')

        self.logger.log('Early exit handler complete - ready for exit')

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
        assert isinstance(x, list)
        assert isinstance(y, list)

        img_batch = x[0]
        mask_batch = x[1]

        # Append first mean teacher data and then superpixel data if using both methods
        if self.using_mean_teacher_method:
            x = x + self._get_mean_teacher_extra_batch_data(img_batch, step_index=step_index, validation=validation)

        if self.using_superpixel_method:
            x = x + self._get_superpixel_extra_batch_data(img_batch, step_index=step_index, validation=validation)

        # Apply possible Gaussian Noise to batch data last e.g. teacher model Gaussian Noise requires unnoised data
        if self.using_gaussian_noise and not validation:
            x[0] = img_batch + self._get_batch_gaussian_noise(noise_shape=img_batch.shape, step_index=step_index)

        # If we are in debug mode, save the batch images - this is right before the images enter
        # into the neural network
        if settings.DEBUG:
            b_min = np.min(img_batch)
            b_max = np.max(img_batch)

            for i in range(0, len(img_batch)):
                img = ((img_batch[i] - b_min) / (b_max - b_min)) * 255.0
                mask = mask_batch[i][:, :, np.newaxis]*255.0
                self.logger.debug_log_image(img, '{}_{}_{}_photo.jpg'.format("val" if validation else "tr", step_index, i), scale=False)
                self.logger.debug_log_image(mask, file_name='{}_{}_{}_mask.png'.format("val" if validation else "tr", step_index, i), format='PNG')

        return x, y

    def _get_batch_gaussian_noise(self, noise_shape, step_index):
        if self.using_gaussian_noise:
            stddev = self.data_augmentation_parameters.gaussian_noise_stddev_function(step_index)
            self.logger.debug_log('Generating gaussian noise with stddev: {}'.format(stddev))
            gnoise = np.random.normal(loc=0.0, scale=stddev, size=noise_shape)
            return gnoise
        else:
            return np.zeros(noise_shape)

    def _get_mean_teacher_extra_batch_data(self, img_batch, step_index, validation):
        if not self.using_mean_teacher_method:
            return []

        if self.teacher_model is None:
            raise ValueError('Teacher model is not set, cannot run predictions')

        # First dimension in all of the input data should be the batch size
        batch_size = img_batch.shape[0]

        if validation:
            # BxHxWxC
            mean_teacher_predictions = np.zeros(shape=(img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], self.num_classes))
            np_consistency_coefficients = np.zeros(shape=[batch_size])
            return [mean_teacher_predictions, np_consistency_coefficients]
        else:
            s_time = time.time()

            if self.using_gaussian_noise:
                teacher_img_batch = img_batch + self._get_batch_gaussian_noise(noise_shape=img_batch.shape, step_index=step_index)
            else:
                teacher_img_batch = img_batch

            # Note: include the training phase noise and dropout layers on the prediction
            mean_teacher_predictions = self.teacher_model.predict_on_batch(teacher_img_batch, use_training_phase_layers=True)
            self.logger.profile_log('Mean teacher batch predictions took: {} s'.format(time.time() - s_time))
            consistency_coefficient = self.consistency_cost_coefficient_function(step_index)
            np_consistency_coefficients = np.ones(shape=[batch_size]) * consistency_coefficient
            return [mean_teacher_predictions, np_consistency_coefficients]

    def _get_superpixel_extra_batch_data(self, img_batch, step_index, validation):
        if not self.using_superpixel_method:
            return []

        # First dimension in all of the input data should be the batch size
        batch_size = img_batch.shape[0]

        if validation:
            np_unlabeled_cost_coefficients = np.zeros(shape=[batch_size])
            return [np_unlabeled_cost_coefficients]
        else:
            unlabeled_cost_coefficient = self.unlabeled_cost_coefficient_function(step_index)
            np_unlabeled_cost_coefficients = np.ones(shape=[batch_size]) * unlabeled_cost_coefficient
            return [np_unlabeled_cost_coefficients]

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

        super(SegmentationTrainer, self).on_batch_end(step_index)

        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot run EMA update')

            a = self.ema_smoothing_coefficient_function(step_index)

            # Perform the EMA weight update: theta'_t = a * theta'_t-1 + (1 - a) * theta_t
            t_weights = self.teacher_model.get_weights()
            s_weights = self.model.get_weights()

            if len(t_weights) != len(s_weights):
                raise ValueError('The weight arrays are not of the same length for the student and teacher: {} vs {}'
                                 .format(len(t_weights), len(s_weights)))

            num_weights = len(t_weights)
            s_time = time.time()

            for i in range(0, num_weights):
                t_weights[i] = a * t_weights[i] + (1.0 - a) * s_weights[i]

            self.teacher_model.set_weights(t_weights)
            self.logger.profile_log('Mean teacher weight update took: {} s'.format(time.time()-s_time))

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

        super(SegmentationTrainer, self).on_epoch_end(epoch_index, step_index, logs)

        if self.using_mean_teacher_method:
            if self.teacher_model is None:
                raise ValueError('Teacher model is not set, cannot validate/save weights')

            # Default to -1.0 validation loss if nothing else is given
            val_loss = -1.0

            if self.teacher_validation_data_generator is not None:
                # Evaluate the mean teacher on the validation data
                validation_steps_per_epoch = self.teacher_validation_data_generator.num_steps_per_epoch

                val_outs = self.teacher_model.evaluate_generator(
                    generator=self.teacher_validation_data_generator,
                    steps=validation_steps_per_epoch if not settings.PROFILE else settings.PROFILE_STEPS_PER_EPOCH,
                    workers=dataset_utils.get_number_of_parallel_jobs())

                val_loss = val_outs[0]
                self.logger.log('Epoch {}: Teacher model val_loss: {}'.format(epoch_index, val_loss))

            self.logger.log('Epoch {}: EMA coefficient {}, consistency cost coefficient: {}'
                            .format(epoch_index, self.ema_smoothing_coefficient_function(step_index), self.consistency_cost_coefficient_function(step_index)))
            self.save_teacher_model_weights(epoch_index=epoch_index, val_loss=val_loss)

    def save_student_model_weights(self, epoch_index, val_loss, file_extension='.student'):
        file_path = self.model_checkpoint_file_path.format(model_folder=self.model_folder_name, epoch=epoch_index, val_loss=val_loss) + file_extension

        # Make sure the directory exists
        general_utils.create_path_if_not_existing(file_path)

        self.logger.log('Saving student model weights to file: {}'.format(file_path))
        self.model.save_weights(file_path, overwrite=True)

    def save_teacher_model_weights(self, epoch_index, val_loss, file_extension='.teacher'):
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

            file_path = teacher_model_checkpoint_file_path.format(model_folder=self.model_folder_name, epoch=epoch_index, val_loss=val_loss) + file_extension

            # Make sure the directory exists
            general_utils.create_path_if_not_existing(file_path)

            self.logger.log('Saving mean teacher model weights to file: {}'.format(file_path))
            self.teacher_model.save_weights(file_path, overwrite=True)

    def get_model_lambda_loss_type(self):
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

    def get_label_generation_function(self, label_generation_function_name):
        # type: (str) -> function

        if label_generation_function_name.lower() == 'felzenszwalb':
            return lambda np_img: image_utils.np_get_felzenswalb_segmentation(np_img, scale=700, sigma=0.6, min_size=250, normalize_img=True, borders_only=True)
        elif label_generation_function_name.lower() == 'slic':
            return lambda np_img: image_utils.np_get_slic_segmentation(np_img, n_segments=300, sigma=1, compactness=10.0, max_iter=20, normalize_img=True, borders_only=True)
        elif label_generation_function_name.lower() == 'quickshift':
            return lambda np_img: image_utils.np_get_quickshift_segmentation(np_img, kernel_size=20, max_dist=15, ratio=0.5, normalize_img=True, borders_only=True)
        elif label_generation_function_name.lower() == 'watershed':
            return lambda np_img: image_utils.np_get_watershed_segmentation(np_img, markers=250, compactness=0.001, normalize_img=True, borders_only=True)
        else:
            raise ValueError('Unknown label generation function name: {}'.format(label_generation_function_name))


#############################################
# CLASSIFICATION TRAINER
#############################################
#
# class ClassificationTrainer(TrainerBase):
#
#     def __init__(self, trainer_type, model_name, model_folder_name, config_file_path, debug=None):
#
#         # Declare variables that are going to be initialized in the _init_ functions
#         self.consistency_cost_coefficient_function = None
#         self.ema_smoothing_coefficient_function = None
#
#         self.material_class_information = None
#         self.data_set_information = None
#         self.num_classes = -1
#
#         self.labeled_photo_files = None
#         self.labeled_mask_files = None
#         self.unlabeled_photo_files = None
#
#         self.training_set_labeled = None
#         self.training_set_unlabeled = None
#         self.validation_set = None
#         self.test_set = None
#
#         self.use_mean_teacher_method = False
#         self.model_wrapper = None
#         self.model = None
#         self.teacher_model_wrapper = None
#         self.teacher_model = None
#         self.initial_epoch = 0
#
#         self.training_data_generator = None
#         self.validation_data_generator = None
#         self.teacher_validation_data_generator = None
#
#         super(ClassificationTrainer, self).__init__(trainer_type, model_name, model_folder_name, config_file_path, debug)
#
#     def _init_config(self):
#         super(ClassificationTrainer, self)._init_config()
#
#         self.minc_classification_data_set = self.get_config_value('minc_classification_data_set')
#
#         if self.minc_classification_data_set is None:
#             raise ValueError('Could not find MINC classification data set from config file with key \'minc_classification_data_set\'')
#
#         self.num_classes = self.minc_classification_data_set['num_classes']
#
#         self.path_to_labeled_photos = self.get_config_value('path_to_labeled_photos')
#         self.use_class_weights = self.get_config_value('use_class_weights')
#
#         self.input_shape = self.get_config_value('input_shape')
#         self.continue_from_last_checkpoint = bool(self.get_config_value('continue_from_last_checkpoint'))
#         self.weights_directory_path = os.path.dirname(self.get_config_value('keras_model_checkpoint_file_path')).format(model_folder=self.model_folder_name)
#         self.use_transfer_weights = bool(self.get_config_value('transfer_weights'))
#         self.transfer_options = self.get_config_value('transfer_options')
#         self.continue_from_optimizer_checkpoint = bool(self.get_config_value('continue_from_optimizer_checkpoint'))
#         self.loss_function_name = self.get_config_value('loss_function')
#
#         # If using mean teacher method read the parameters - any missing parameters should raise
#         # exceptions
#         self.use_mean_teacher_method = bool(self.get_config_value('use_mean_teacher_method'))
#
#         if self.use_mean_teacher_method:
#             self.logger.log('Reading mean teacher method configuration')
#             mean_teacher_params = self.get_config_value('mean_teacher_params')
#
#             if mean_teacher_params is None:
#                 raise ValueError('Could not find entry for mean_teacher_params from the configuration JSON')
#
#             teacher_weights_directory_path = os.path.dirname(mean_teacher_params['teacher_model_checkpoint_file_path']).format(model_folder=self.model_folder_name)
#             self.logger.log('Teacher weights directory path: {}'.format(teacher_weights_directory_path))
#             self.teacher_weights_directory_path = teacher_weights_directory_path
#             self.teacher_model_checkpoint_file_path = mean_teacher_params['teacher_model_checkpoint_file_path']
#
#             ema_coefficient_schedule_function = mean_teacher_params['ema_smoothing_coefficient_function']
#             self.logger.log('EMA smoothing coefficient function: {}'.format(ema_coefficient_schedule_function))
#             self.ema_smoothing_coefficient_function = eval(ema_coefficient_schedule_function)
#
#             consistency_cost_coefficient_function = mean_teacher_params['consistency_cost_coefficient_function']
#             self.logger.log('Consistency cost coefficient function: {}'.format(consistency_cost_coefficient_function))
#             self.consistency_cost_coefficient_function = eval(consistency_cost_coefficient_function)
#
#         self.use_data_augmentation = bool(self.get_config_value('use_data_augmentation'))
#         self.num_color_channels = self.get_config_value('num_color_channels')
#         self.random_seed = self.get_config_value('random_seed')
#
#         self.num_epochs = self.get_config_value('num_epochs')
#         self.num_labeled_per_batch = self.get_config_value('num_labeled_per_batch')
#         self.crop_shape = self.get_config_value('crop_shape')
#         self.resize_shape = self.get_config_value('resize_shape')
#         self.validation_num_labeled_per_batch = self.get_config_value('validation_num_labeled_per_batch')
#         self.validation_crop_shape = self.get_config_value('validation_crop_shape')
#         self.validation_resize_shape = self.get_config_value('validation_resize_shape')
#
#     def _init_data(self):
#         super(ClassificationTrainer, self)._init_data()
#
#         # Initialize class weights
#         if self.get_config_value('use_class_weights'):
#             override_class_weights = self.get_config_value('override_class_weights')
#
#             if override_class_weights is None:
#                 raise ValueError('Use class weights is true, but override class weights could not be found. ClassificationTrainer only supports override class weights.')
#
#             if override_class_weights is not None:
#                 self.logger.log('Found override class weights')
#                 self.class_weights = np.array(override_class_weights)
#         else:
#             self.class_weights = np.ones(np.ones([self.num_classes], dtype=np.float32))
#
#         self.logger.log('Using class weights: {}'.format(self.class_weights))
#
#     def _init_models(self):
#         super(ClassificationTrainer, self)._init_models()
#
#         # Are we using the mean teacher method or superpixel method?
#         self.logger.log('Use mean teacher method: {}'.format(self.use_mean_teacher_method))
#
#         # Model creation
#         student_model_type = self.get_model_type()
#
#         self.logger.log('Creating student model {} instance with type: {}, input shape: {}, num classes: {}'
#                  .format(self.model_name, student_model_type, self.input_shape, self.num_classes))
#
#         self.model_wrapper = get_model(self.model_name,
#                                        self.input_shape,
#                                        self.num_classes,
#                                        model_lambda_loss_type=student_model_type)
#
#         self.model = self.model_wrapper.model
#         self.model.summary()
#
#         if self.continue_from_last_checkpoint:
#             self.initial_epoch = self.load_latest_weights_for_model(self.model, self.weights_directory_path)
#
#         if self.use_transfer_weights:
#             if self.initial_epoch != 0:
#                 self.logger.log('Cannot transfer weights when continuing from last checkpoint. Skipping weight transfer')
#             else:
#                 self.transfer_weights(self.model_wrapper, self.transfer_options)
#
#         # Get the optimizer for the model
#         if self.continue_from_optimizer_checkpoint and self.initial_epoch == 0:
#             self.logger.log('Cannot continue from optimizer checkpoint if initial epoch is 0. Ignoring optimizer checkpoint.')
#             self.continue_from_optimizer_checkpoint = False
#
#         optimizer = self.get_optimizer(self.continue_from_optimizer_checkpoint)
#
#         # Get the loss function for the student model
#         if self.use_mean_teacher_method and self.loss_function_name != 'dummy':
#             self.logger.log('In Mean Teacher mode trainer should use \'dummy\' loss function, got: {}. Ignoring passed loss function.'.format(self.loss_function_name))
#             self.loss_function_name = 'dummy'
#
#         loss_function = self.get_loss_function(self.loss_function_name,
#                                                use_class_weights=self.use_class_weights,
#                                                class_weights=self.class_weights,
#                                                num_classes=self.num_classes)
#
#         # Compile the student model
#         if self.use_mean_teacher_method:
#             self.model.compile(optimizer=optimizer,
#                                loss={'loss': loss_function, 'logits': lambda _, y_pred: 0.0*y_pred},
#                                loss_weights={'loss': 1., 'logits': 0.},
#                                metrics={'logits': ['accuracy']},
#                                **self.get_compile_kwargs())
#         else:
#             self.model.compile(optimizer=optimizer,
#                                loss=loss_function,
#                                metrics=['accuracy'],
#                                **self.get_compile_kwargs())
#
#         # If we are using the mean teacher method create the teacher model
#         if self.use_mean_teacher_method:
#             teacher_model_type = ModelLambdaLossType.MEAN_TEACHER_TEACHER
#             self.logger.log('Creating teacher model {} instance with type: {}, input shape: {}, num classes: {}'.format(self.model_name, teacher_model_type, self.input_shape, self.num_classes))
#             self.teacher_model_wrapper = get_model(self.model_name, self.input_shape, self.num_classes, model_lambda_loss_type=teacher_model_type)
#             self.teacher_model = self.teacher_model_wrapper.model
#             self.teacher_model.summary()
#
#             if self.continue_from_last_checkpoint:
#                 self.logger.log('Loading latest teacher model weights from path: {}'.format(self.teacher_weights_directory_path))
#                 initial_teacher_epoch = self.load_latest_weights_for_model(self.teacher_model, self.teacher_weights_directory_path)
#
#                 if initial_teacher_epoch < 1:
#                     self.logger.log('Could not find suitable weights, initializing teacher with student model weights')
#                     self.teacher_model.set_weights(self.model.get_weights())
#             else:
#                 self.teacher_model.set_weights(self.model.get_weights())
#
#             teacher_class_weights = self.class_weights
#
#             # Note: Teacher model can use the regular metrics
#             self.teacher_model.compile(optimizer=optimizer,
#                                        loss=losses.categorical_crossentropy_loss(teacher_class_weights),
#                                        metrics=['accuracy'],
#                                        **self.get_compile_kwargs())
#
#     def _init_data_generators(self):
#         super(ClassificationTrainer, self)._init_data_generators()
#
#         # Create training data and validation data generators
#         # Note: training data comes from semi-supervised segmentation data generator and validation
#         # and test data come from regular segmentation data generator
#         self.logger.log('Creating training data generator')
#
#         training_data_generator_params = DataGeneratorParameters(
#             material_class_information=None,
#             num_color_channels=self.num_color_channels,
#             random_seed=self.random_seed,
#             crop_shape=self.crop_shape,
#             resize_shape=self.resize_shape,
#             use_per_channel_mean_normalization=True,
#             per_channel_mean=self.minc_classification_data_set.get('per_channel_mean'),
#             use_per_channel_stddev_normalization=True,
#             per_channel_stddev=self.minc_classification_data_set.get('per_channel_stddev'),
#             use_data_augmentation=self.use_data_augmentation,
#             use_material_samples=False,
#             data_augmentation_params=self.data_augmentation_parameters,
#             shuffle_data_after_epoch=True)
#
#         self.training_data_generator = MINCClassificationDataGenerator(
#             minc_data_set_file_path=self.minc_classification_data_set.get('training_set_file_path'),
#             minc_labels_translation_file_path=self.minc_classification_data_set.get('labels_translation_file_path'),
#             minc_photos_folder_path=self.path_to_labeled_photos,
#             num_labeled_per_batch=self.num_labeled_per_batch,
#             params=training_data_generator_params)
#
#         self.logger.log('Creating validation data generator')
#
#         validation_data_generator_params = DataGeneratorParameters(
#             material_class_information=None,
#             num_color_channels=self.num_color_channels,
#             random_seed=self.random_seed,
#             crop_shape=self.validation_crop_shape,
#             resize_shape=self.validation_resize_shape,
#             use_per_channel_mean_normalization=True,
#             per_channel_mean=training_data_generator_params.per_channel_mean,
#             use_per_channel_stddev_normalization=True,
#             per_channel_stddev=training_data_generator_params.per_channel_stddev,
#             use_data_augmentation=False,
#             use_material_samples=False,
#             data_augmentation_params=None,
#             shuffle_data_after_epoch=True)
#
#         # The student lambda loss layer needs semi supervised input, so we need to work around it
#         # to only provide labeled input from the semi-supervised data generator. The dummy data
#         # is appended to each batch so that the batch data maintains it's shape. This is done in the
#         # modify_batch_data function.
#         self.validation_data_generator = MINCClassificationDataGenerator(
#             minc_data_set_file_path=self.minc_classification_data_set.get('validation_set_file_path'),
#             minc_labels_translation_file_path=self.minc_classification_data_set.get('labels_translation_file_path'),
#             minc_photos_folder_path=self.path_to_labeled_photos,
#             num_labeled_per_batch=self.validation_num_labeled_per_batch,
#             params=validation_data_generator_params)
#
#         # Note: The teacher has a regular SegmentationDataGenerator for validation data generation
#         # because it doesn't have the semi supervised loss lambda layer
#         if self.use_mean_teacher_method:
#             self.teacher_validation_data_generator = MINCClassificationDataGenerator(
#                 minc_data_set_file_path=self.minc_classification_data_set.get('validation_set_file_path'),
#                 minc_labels_translation_file_path=self.minc_classification_data_set.get('labels_translation_file_path'),
#                 minc_photos_folder_path=self.path_to_labeled_photos,
#                 num_labeled_per_batch=self.validation_num_labeled_per_batch,
#                 params=validation_data_generator_params)
#         else:
#             self.teacher_validation_data_generator = None
#
#         self.logger.log('Using per-channel mean: {}'.format(self.training_data_generator.per_channel_mean))
#         self.logger.log('Using per-channel stddev: {}'.format(self.training_data_generator.per_channel_stddev))
#
#     def train(self):
#         super(ClassificationTrainer, self).train()
#
#         total_batch_size = self.num_labeled_per_batch
#         training_steps_per_epoch = self.training_data_generator.num_steps_per_epoch
#         validation_steps_per_epoch = self.validation_data_generator.num_steps_per_epoch
#         num_workers = dataset_utils.get_number_of_parallel_jobs()
#
#         self.logger.log('Num epochs: {}, initial epoch: {}, total batch size: {}, crop shape: {}, training steps per epoch: {}, validation steps per epoch: {}'
#                  .format(self.num_epochs, self.initial_epoch, total_batch_size, self.crop_shape, training_steps_per_epoch, validation_steps_per_epoch))
#
#         self.logger.log('Num workers: {}'.format(num_workers))
#
#         # Get a list of callbacks
#         callbacks = self.get_callbacks()
#
#         # Sanity check
#         if not isinstance(self.model, ExtendedModel):
#             raise ValueError('When using classification training the model must be an instance of ExtendedModel')
#
#         # Note: the student model should not be evaluated using the validation data generator
#         # the generator input will not be meaning
#         self.model.fit_generator(
#             generator=self.training_data_generator,
#             steps_per_epoch=training_steps_per_epoch if not self.debug else self.debug_steps_per_epoch,
#             epochs=self.num_epochs if not self.debug else self.debug_num_epochs,
#             initial_epoch=self.initial_epoch,
#             validation_data=self.validation_data_generator,
#             validation_steps=validation_steps_per_epoch if not self.debug else self.debug_steps_per_epoch,
#             verbose=1,
#             trainer=self,
#             callbacks=callbacks,
#             workers=num_workers,
#             debug=self.debug is not None)
#
#         if self.debug:
#             self.logger.log('Saving debug data to: {}'.format(self.debug))
#             self.debug_timeliner.save(self.debug)
#
#         self.logger.log('The training session ended at local time {}\n'.format(datetime.datetime.now()))
#         self.logger.log_file.close()
#
#     def handle_early_exit(self):
#         super(ClassificationTrainer, self).handle_early_exit()
#
#         if not self.save_values_on_early_exit:
#             return
#
#         # Stop training
#         self.logger.log('Stopping student model training')
#         self.model.stop_fit_generator()
#
#         # Save student model weights
#         self.logger.log('Saving student model weights')
#         self.save_student_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.student-early-stop')
#
#         # Save teacher model weights
#         if self.use_mean_teacher_method:
#             self.logger.log('Saving teacher model weights')
#             self.save_teacher_model_weights(epoch_index=self.last_completed_epoch, val_loss=-1.0, file_extension='.teacher-early-stop')
#
#         # Save optimizer settings
#         self.logger.log('Saving optimizer settings')
#         self.save_optimizer_settings(model=self.model, file_extension='.student-early-stop')
#
#         self.logger.log('Early exit handler complete - ready for exit')
#
#     def modify_batch_data(self, step_index, x, y, validation=False):
#         # type: (int, list[np.array[np.float32]], np.array, bool) -> (list[np.array[np.float32]], np.array)
#
#         """
#         Invoked by the ExtendedModel right before train_on_batch:
#
#         If using the Mean Teacher method:
#
#             Modifies the batch data by appending the mean teacher predictions as the last
#             element of the input data X if we are using mean teacher training.
#
#         # Arguments
#             :param step_index: the training step index
#             :param x: input data
#             :param y: output data
#             :param validation: is this a validation data batch
#         # Returns
#             :return: a tuple of (input data, output data)
#         """
#
#         images = x[0]
#         #images = images + np.random.normal(loc=0.0, scale=0.03, size=images.shape)
#         #x[0] = images
#
#         if self.model_wrapper.model_type == ModelLambdaLossType.MEAN_TEACHER_STUDENT:
#             if self.teacher_model is None:
#                 raise ValueError('Teacher model is not set, cannot run predictions')
#
#             # First dimension in all of the input data should be the batch size
#             images = x[0]
#             batch_size = images.shape[0]
#
#             if validation:
#                 # BxHxWxC
#                 mean_teacher_predictions = np.zeros(shape=(images.shape[0], images.shape[1], images.shape[2], self.num_classes))
#                 np_consistency_coefficients = np.zeros(shape=[batch_size])
#                 x = x + [mean_teacher_predictions, np_consistency_coefficients]
#             else:
#                 s_time = 0
#
#                 if self.debug:
#                     s_time = time.time()
#
#                 mean_teacher_predictions = self.teacher_model.predict_on_batch(images)
#
#                 if self.debug:
#                     self.logger.log('Mean teacher batch predictions took: {} s'.format(time.time()-s_time))
#
#                 consistency_coefficient = self.consistency_cost_coefficient_function(step_index)
#                 np_consistency_coefficients = np.ones(shape=[batch_size]) * consistency_coefficient
#                 x = x + [mean_teacher_predictions, np_consistency_coefficients]
#
#         return x, y
#
#     def on_batch_end(self, step_index):
#         # type: (int) -> ()
#
#         """
#         Invoked by the ExtendedModel right after train_on_batch:
#
#         Updates the teacher model weights if using the mean teacher method for
#         training, otherwise does nothing.
#
#         # Arguments
#             :param step_index: the training step index
#         # Returns
#             Nothing
#         """
#
#         super(ClassificationTrainer, self).on_batch_end(step_index)
#
#         if self.use_mean_teacher_method:
#             if self.teacher_model is None:
#                 raise ValueError('Teacher model is not set, cannot run EMA update')
#
#             a = self.ema_smoothing_coefficient_function(step_index)
#
#             # Perform the EMA weight update: theta'_t = a * theta'_t-1 + (1 - a) * theta_t
#             t_weights = self.teacher_model.get_weights()
#             s_weights = self.model.get_weights()
#
#             if len(t_weights) != len(s_weights):
#                 raise ValueError('The weight arrays are not of the same length for the student and teacher: {} vs {}'
#                                  .format(len(t_weights), len(s_weights)))
#
#             num_weights = len(t_weights)
#             s_time = 0
#
#             if self.debug:
#                 s_time = time.time()
#
#             for i in range(0, num_weights):
#                 t_weights[i] = a * t_weights[i] + (1.0 - a) * s_weights[i]
#
#             self.teacher_model.set_weights(t_weights)
#
#             if self.debug:
#                 self.logger.log('Mean teacher weight update took: {} s'.format(time.time()-s_time))
#
#     def on_epoch_end(self, epoch_index, step_index, logs):
#         # type: (int, int, dict) -> ()
#
#         """
#         Invoked by the ExtendedModel right after the epoch is over.
#
#         Evaluates mean teacher model on the validation data and saves the mean teacher
#         model weights.
#
#         # Arguments
#             :param epoch_index: index of the epoch that has finished
#             :param step_index: index of the step that has finished
#             :param logs: logs from the epoch (for the student model)
#         # Returns
#             Nothing
#         """
#
#         super(ClassificationTrainer, self).on_epoch_end(epoch_index, step_index, logs)
#
#         if self.use_mean_teacher_method:
#             if self.teacher_model is None:
#                 raise ValueError('Teacher model is not set, cannot validate/save weights')
#
#             # Default to -1.0 validation loss if nothing else is given
#             val_loss = -1.0
#
#             if self.teacher_validation_data_generator is not None:
#                 # Evaluate the mean teacher on the validation data
#                 validation_steps_per_epoch = dataset_utils.get_number_of_batches(
#                     self.validation_set.size,
#                     self.validation_num_labeled_per_batch)
#
#                 val_outs = self.teacher_model.evaluate_generator(
#                     generator=self.teacher_validation_data_generator,
#                     steps=validation_steps_per_epoch if not self.debug else self.debug_steps_per_epoch,
#                     workers=dataset_utils.get_number_of_parallel_jobs())
#
#                 val_loss = val_outs[0]
#                 self.logger.log('\nEpoch {}: Mean teacher validation loss {}'.format(epoch_index, val_loss))
#
#             self.logger.log('\nEpoch {}: EMA coefficient {}, consistency cost coefficient: {}'
#                      .format(epoch_index, self.ema_smoothing_coefficient_function(step_index), self.consistency_cost_coefficient_function(step_index)))
#             self.save_teacher_model_weights(epoch_index=epoch_index, val_loss=val_loss)
#
#     def save_student_model_weights(self, epoch_index, val_loss, file_extension='.student'):
#         file_path = self.get_config_value('keras_model_checkpoint_file_path')\
#                         .format(model_folder=self.model_folder_name, epoch=epoch_index, val_loss=val_loss) + file_extension
#
#         # Make sure the directory exists
#         TrainerBase._create_path_if_not_existing(file_path)
#
#         self.logger.log('Saving student model weights to file: {}'.format(file_path))
#         self.model.save_weights(file_path, overwrite=True)
#
#     def save_teacher_model_weights(self, epoch_index, val_loss, file_extension='.teacher'):
#         if self.use_mean_teacher_method:
#             if self.teacher_model is None:
#                 raise ValueError('Teacher model is not set, cannot save weights')
#
#             # Save the teacher model weights:
#             teacher_model_checkpoint_file_path = self.teacher_model_checkpoint_file_path
#
#             # Don't crash here, too much effort done - save with a different name to the same path as
#             # the student model
#             if teacher_model_checkpoint_file_path is None:
#                 self.logger.log('Value of teacher_model_checkpoint_file_path is not set - defaulting to teacher folder under student directory')
#                 file_name_format = os.path.basename(self.get_config_value('keras_model_checkpoint_file_path'))
#                 teacher_model_checkpoint_file_path = os.path.join(os.path.join(self.weights_directory_path, 'teacher/'), file_name_format)
#
#             file_path = teacher_model_checkpoint_file_path.format(model_folder=self.model_folder_name, epoch=epoch_index, val_loss=val_loss) + file_extension
#
#             # Make sure the directory exists
#             TrainerBase._create_path_if_not_existing(file_path)
#
#             self.logger.log('Saving mean teacher model weights to file: {}'.format(file_path))
#             self.teacher_model.save_weights(file_path, overwrite=True)
#
#     def get_model_type(self):
#         if self.use_mean_teacher_method:
#             return ModelLambdaLossType.MEAN_TEACHER_STUDENT_CLASSIFICATION
#         else:
#             return ModelLambdaLossType.NONE
