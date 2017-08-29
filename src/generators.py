# coding=utf-8

import random
import math
import numpy as np
import threading
from enum import Enum
from typing import Callable

from keras.preprocessing.image import img_to_array
from PIL import Image

from utils import dataset_utils
from utils import image_utils
from utils.image_utils import ImageInterpolation, ImageTransform
from utils.dataset_utils import MaterialClassInformation, MaterialSample, MINCSample
from data_set import LabeledImageDataSet, UnlabeledImageDataSet, ImageFile, ImageSet
import settings

from abc import ABCMeta, abstractmethod, abstractproperty
import keras.backend as K


#######################################
# UTILITY CLASSES
#######################################


class DataAugmentationParameters:
    """
    This class helps to maintain the data augmentation parameters for data generators.
    """

    def __init__(self,
                 augmentation_probability_function,
                 rotation_range=0.,
                 zoom_range=0.,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 channel_shift_range=None,
                 horizontal_flip=False,
                 vertical_flip=False,
                 fill_mode='constant',
                 gaussian_noise_stddev_function=None,
                 gamma_adjust_range=0.0):
        """
        # Arguments
            :param augmentation_probability_function: a schedule lambda function as a string, which takes step index and returns a float in range [0.0, 1.0]
            :param rotation_range: range of random rotations
            :param zoom_range: range of random zooms
            :param width_shift_range: fraction of total width [0, 1]
            :param height_shift_range: fraction of total height [0, 1]
            :param channel_shift_range: channel shift range as a float, enables shifting channels between [-val, val]
            :param horizontal_flip: should horizontal flips be applied
            :param vertical_flip: should vertical flips be applied
            :param fill_mode: how should we fill overgrown areas
            :param gaussian_noise_stddev_function: a schedule lambda function as a string, which takes step index and returns the stddev of the gaussian noise,
            expected to be in range [0,1]
        # Returns
            :return: A new instance of DataAugmentationParameters
        """

        if augmentation_probability_function is None:
            raise ValueError('Augmentation probability function cannot be None - needs to be an evaluatable lambda function string')

        self.augmentation_probability_function = eval(augmentation_probability_function)
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self.gaussian_noise_stddev_function = eval(gaussian_noise_stddev_function) if gaussian_noise_stddev_function is not None else None

        # Zoom range can either be a tuple or a scalar
        if zoom_range is None:
            self.zoom_range = None
        elif np.isscalar(zoom_range):
            self.zoom_range = np.array([1.0 - zoom_range, 1.0 + zoom_range])
        elif len(zoom_range) == 2:
            self.zoom_range = np.array(zoom_range, dtype=np.float32)
        else:
            raise ValueError('zoom_range should be a float or a tuple or list of two floats. Received arg: ', zoom_range)

        # Gamma adjust range can be a tuple or a scalar
        if gamma_adjust_range is None:
            self.gamma_adjust_range = None
        elif np.isscalar(gamma_adjust_range):
            self.gamma_adjust_range = np.array([1.0 - gamma_adjust_range, 1.0 + gamma_adjust_range])
        elif len(gamma_adjust_range) == 2:
            self.gamma_adjust_range = np.array(gamma_adjust_range, dtype=np.float32)
        else:
            raise ValueError('gamma_adjust_range should be a float or a tuple or list of two floats. Received arg: ', gamma_adjust_range)

        if gamma_adjust_range is not None and np.min(self.gamma_adjust_range) < 0.0:
            raise ValueError('Gamma should always be a positive value, now got range: {}'.format(list(gamma_adjust_range)))


class DataGeneratorParameters(object):
    """
    This class helps to maintain parameters in common with all different data generators.
    """

    def __init__(self,
                 material_class_information,
                 num_color_channels,
                 logger,
                 random_seed=None,
                 crop_shape=None,
                 resize_shape=None,
                 use_per_channel_mean_normalization=True,
                 per_channel_mean=None,
                 use_per_channel_stddev_normalization=True,
                 per_channel_stddev=None,
                 photo_cval=None,
                 mask_cval=None,
                 use_data_augmentation=False,
                 data_augmentation_params=None,
                 shuffle_data_after_epoch=True,
                 use_material_samples=False,
                 div2_constraint=4):
        """
        Builds a wrapper for DataGenerator parameters

        # Arguments
            :param material_class_information:
            :param num_color_channels: number of channels in the photos; 1, 3 or 4
            :param logger: a Logger instance for logging text and images
            :param random_seed: an integer random seed
            :param crop_shape: size of the crop or None if no cropping should be applied
            :param resize_shape: size of the desired resized images, None if no resizing should be applied
            :param use_per_channel_mean_normalization: whether per-channel mean normalization should be applied
            :param per_channel_mean: per channel mean in range [-1, 1]
            :param use_per_channel_stddev_normalization: whether per-channel stddev normalizaiton should be applied
            :param per_channel_stddev: per-channel stddev in range [-1, 1]
            :param photo_cval: fill color value for photos [0,255], otherwise per-channel mean used
            :param mask_cval: fill color value for masks [0,255], otherwise [0,0,0] used
            :param use_data_augmentation: should data augmentation be used
            :param data_augmentation_params: parameters for data augmentation
            :param shuffle_data_after_epoch: should the data be shuffled after every epoch
            :param use_material_samples: should material samples be used
            :param div2_constraint: how many times does the image/crop need to be divisible by two
        # Returns
            Nothing
        """

        self.material_class_information = material_class_information
        self.num_color_channels = num_color_channels
        self.logger = logger
        self.random_seed = random_seed
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.use_per_channel_mean_normalization = use_per_channel_mean_normalization
        self.per_channel_mean = per_channel_mean
        self.use_per_channel_stddev_normalization = use_per_channel_stddev_normalization
        self.per_channel_stddev = per_channel_stddev
        self.photo_cval = photo_cval
        self.mask_cval = mask_cval
        self.use_data_augmentation = use_data_augmentation
        self.data_augmentation_params = data_augmentation_params
        self.shuffle_data_after_epoch = shuffle_data_after_epoch
        self.use_material_samples = use_material_samples
        self.div2_constraint = div2_constraint


#######################################
# ITERATOR
#######################################


class DataSetIterator(object):

    __metaclass__ = ABCMeta

    def __init__(self, n, batch_size, shuffle, seed, logger):
        self.n = n
        self.batch_size = min(batch_size, n)    # The batch size could in theory be bigger than the data set size
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger
        self.batch_index = 0                    # The index of the batch within the epoch
        self.step_index = 0                     # The overall index of the batch

    def reset(self):
        self.batch_index = 0
        self.step_index = 0

    @abstractmethod
    def get_next_batch(self):
        if self.seed is not None:
            np.random.seed(self.seed + self.step_index)

    @abstractproperty
    def num_steps_per_epoch(self):
        pass

    def _get_number_of_batches(self, data_set_size, batch_size):
        # type: (int, int) -> int

        """
        Returns the number of batches for the given data set size and batch size.
        The function assumes that all data will be used every epoch and the last batch size
        can be smaller than the others.

        # Arguments
            :param data_set_size: data set size
            :param batch_size: batch size
        # Returns
            :return: the number of batches from this dataset
        """
        num_batches = int(math.ceil(float(data_set_size) / float(batch_size)))
        return num_batches


class BasicDataSetIterator(DataSetIterator):
    """
    A class for iterating over a data set in batches.
    """

    def __init__(self, n, batch_size, shuffle, seed, logger):
        # type: (int, int, bool, int, Logger) -> None

        """
        # Arguments
            :param n: Integer, total number of samples in the dataset to loop over.
            :param batch_size: Integer, size of a batch.
            :param shuffle: Boolean, whether to shuffle the data between epochs.
            :param seed: Random seeding for data shuffling.
            :param logger: Logger instance for logging
        # Returns
            Nothing
        """
        super(BasicDataSetIterator, self).__init__(n=n, batch_size=batch_size, shuffle=shuffle, seed=seed, logger=logger)
        self.index_array = np.arange(self.n)

    def get_next_batch(self):
        # type: () -> (np.array[int], int, int)

        super(BasicDataSetIterator, self).get_next_batch()

        if self.batch_index == 0:
            self.index_array = np.arange(self.n)
            if self.shuffle:
                self.index_array = np.random.permutation(self.n)

        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:
            current_batch_size = self.batch_size
            self.batch_index += 1
        else:
            current_batch_size = self.n - current_index
            self.batch_index = 0

        self.step_index += 1

        return self.index_array[current_index: current_index + current_batch_size], current_index, current_batch_size, self.step_index

    @property
    def num_steps_per_epoch(self):
        return self._get_number_of_batches(self.n, self.batch_size)


class MaterialSampleIterationMode(Enum):
    UNIFORM_MAX = 0     # Sample each material class uniformly. Set the number of steps per epoch according to max class num samples.
    UNIFORM_MIN = 1     # Sample each material class uniformly. Set the number of steps per epoch according to min class num samples.
    UNIFORM_MEAN = 2    # Sample each class uniformly. Set the number of steps per epoch according to mean samples per class.
    UNIQUE = 3          # Iterate through all the unique samples once within epoch - means no balancing


class MaterialSampleDataSetIterator(DataSetIterator):
    """
    A class for iterating randomly through MaterialSamples for a data set in batches.
    """

    def __init__(self, material_samples, batch_size, shuffle, seed, logger, iter_mode=MaterialSampleIterationMode.UNIFORM_MAX):
        # type: (list[list[MaterialSample]], int, bool, int, Logger, MaterialSampleIterationMode) -> None

        self._num_unique_material_samples = sum(len(material_category) for material_category in material_samples)
        super(MaterialSampleDataSetIterator, self).__init__(n=self._num_unique_material_samples, batch_size=batch_size, shuffle=shuffle, seed=seed, logger=logger)

        # Calculate uniform probabilities for all classes that have non zero samples
        self.iter_mode = iter_mode
        self._material_category_sampling_probabilities = [0.0] * len(material_samples)
        self._num_non_zero_classes = sum(1 for material_category in material_samples if len(material_category) > 0)

        self.logger.debug_log('Samples per material category: {}'.format([len(material_category) for material_category in material_samples]))

        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            self._samples_per_material_category_per_epoch = None
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MAX:
            self._samples_per_material_category_per_epoch = max([len(material_category) for material_category in material_samples])
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MIN:
            self._samples_per_material_category_per_epoch = min([len(material_category) for material_category in material_samples])
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MEAN:
            self._samples_per_material_category_per_epoch = int(np.mean(np.array([len(material_category) for material_category in material_samples])))
        else:
            raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

        # Build index lists for the different material samples
        self._material_samples = []

        for material_category in material_samples:
            if not shuffle:
                self._material_samples.append(np.arange(len(material_category)))
            else:
                self._material_samples.append(np.random.permutation(len(material_category)))

        # Build a flattened list of all the material samples (for unique iteration)
        self._material_samples_flattened = []

        for i in range(len(self._material_samples)):
            for j in range(len(self._material_samples[i])):
                self._material_samples_flattened.append((i, j))

        # Calculate the sampling probabilities for each class - for uniform sampling all non-zero material
        # categories should have the same probabilities
        material_category_sampling_probability = 1.0 / self._num_non_zero_classes

        for i in range(len(material_samples)):
            num_samples_in_category = len(material_samples[i])

            if num_samples_in_category > 0:
                self._material_category_sampling_probabilities[i] = material_category_sampling_probability
            else:
                # Zero is assumed as the background class and should/can have zero instances
                if i != 0:
                    self.logger.warn('Material class {} has 0 material samples'.format(i))

                self._material_category_sampling_probabilities[i] = 0.0

        self.logger.debug_log('Material category sampling probabilities: {}'.format(self._material_category_sampling_probabilities))

        # Keep track of the current sample (next sample to be given) in each material category
        self.num_material_classes = len(material_samples)
        self._current_samples = [0] * self.num_material_classes

    def get_next_batch(self):
        # type: () -> (list[tuple[int]], int, int)

        """
        Gives the next batch as tuple of indices to the material samples 2D array.
        The tuples are in the form (sample_category_idx, sample_idx).

        # Arguments
            Nothing
        # Returns
            :return: a batch of material samples as tuples of indices into 2D material sample array
        """

        super(MaterialSampleDataSetIterator, self).get_next_batch()

        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            return self._get_next_batch_unique()
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MAX or self.iter_mode == MaterialSampleIterationMode.UNIFORM_MIN or self.iter_mode == MaterialSampleIterationMode.UNIFORM_MEAN:
            return self._get_next_batch_uniform()

        raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

    def _get_next_batch_uniform(self):
        sample_categories = np.random.choice(a=self.num_material_classes, size=self.batch_size, p=self._material_category_sampling_probabilities)
        batch = []

        for sample_category_idx in sample_categories:
            internal_sample_idx = self._current_samples[sample_category_idx]
            sample_idx = self._material_samples[sample_category_idx][internal_sample_idx]
            batch.append((sample_category_idx, sample_idx))

            # Keep track of the used samples in each category
            self._current_samples[sample_category_idx] += 1

            # If all of the samples in the category have been used, zero out the
            # index for the category and shuffle the category list if shuffle is enabled
            if self._current_samples[sample_category_idx] >= len(self._material_samples[sample_category_idx]):
                self._current_samples[sample_category_idx] = 0

                if self.shuffle:
                    self.logger.debug_log('Shuffling material sample category {} with {} samples'
                                          .format(sample_category_idx, len(self._material_samples[sample_category_idx])))
                    self._material_samples[sample_category_idx] = np.random.permutation(len(self._material_samples[sample_category_idx]))
                else:
                    self._material_samples[sample_category_idx] = np.arange(len(self._material_samples[sample_category_idx]))

        n_samples = self._samples_per_material_category_per_epoch * self._num_non_zero_classes
        current_index = (self.batch_index * self.batch_size) % n_samples
        current_batch_size = len(batch)

        if n_samples > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        self.step_index += 1

        self.logger.debug_log('Batch {}: {}'.format(self.step_index, batch))
        return batch, current_index, current_batch_size, self.step_index

    def _get_next_batch_unique(self):
        if self.batch_index == 0 and self.shuffle:
            np.random.shuffle(self._material_samples_flattened)

        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:
            current_batch_size = self.batch_size
            self.batch_index += 1
        else:
            current_batch_size = self.n - current_index
            self.batch_index = 0

        self.step_index += 1

        batch = self._material_samples_flattened[current_index: current_index + current_batch_size]
        self.logger.debug_log('Batch {}: {}'.format(self.step_index, list(batch)))
        return batch, current_index, current_batch_size, self.step_index

    @property
    def num_steps_per_epoch(self):
        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            return self._get_number_of_batches(self._num_unique_material_samples, self.batch_size)
        # If all classes are sampled uniformly, we have been through all the samples in the data
        # On average after we have gone through all the samples in the largest class, but min and mean are also valid
        else:
            return self._get_number_of_batches(self._samples_per_material_category_per_epoch * self._num_non_zero_classes, self.batch_size)


#######################################
# DATA GENERATOR
#######################################


class SegmentationMaskEncodingType(Enum):
    INDEX = 0
    ONE_HOT = 1


class BatchDataFormat(Enum):
    SUPERVISED = 0
    SEMI_SUPERVISED = 1


class DataGenerator(object):

    """
    Abstract class which declares necessary methods for different DataGenerators. Also,
    unwraps the DataGeneratorParameters to class member variables.
    """

    __metaclass__ = ABCMeta

    def __init__(self, batch_data_format, params):
        # type: (BatchDataFormat, DataGeneratorParameters) -> None

        self.lock = threading.Lock()
        self.batch_data_format = batch_data_format

        # Unwrap DataGeneratorParameters to member variables
        self.material_class_information = params.material_class_information
        self.num_color_channels = params.num_color_channels
        self.logger = params.logger
        self.random_seed = params.random_seed
        self.crop_shape = params.crop_shape
        self.resize_shape = params.resize_shape
        self.use_per_channel_mean_normalization = params.use_per_channel_mean_normalization
        self.per_channel_mean = params.per_channel_mean
        self.use_per_channel_stddev_normalization = params.use_per_channel_stddev_normalization
        self.per_channel_stddev = params.per_channel_stddev
        self.photo_cval = params.photo_cval
        self.mask_cval =params.mask_cval
        self.use_data_augmentation = params.use_data_augmentation
        self.data_augmentation_params = params.data_augmentation_params
        self.shuffle_data_after_epoch = params.shuffle_data_after_epoch
        self.use_material_samples = params.use_material_samples
        self.div2_constraint = params.div2_constraint

        # Other member variables
        self.img_data_format = K.image_data_format()

        # Ensure the per_channel_mean is a numpy tensor
        if self.per_channel_mean is not None:
            self.per_channel_mean = np.array(self.per_channel_mean, dtype=np.float32)

        # Ensure per_channel_stddev is a numpy tensor
        if self.per_channel_stddev is not None:
            self.per_channel_stddev = np.array(self.per_channel_stddev, dtype=np.float32)

        # Calculate missing per-channel mean if necessary
        if self.use_per_channel_mean_normalization and (self.per_channel_mean is None or len(self.per_channel_mean) != self.num_color_channels):
            self.per_channel_mean = np.array(dataset_utils.calculate_per_channel_mean(self.get_all_photos(), self.num_color_channels))
            self.logger.log('DataGenerator: Using per-channel mean: {}'.format(list(self.per_channel_mean)))

        # Calculate missing per-channel stddev if necessary
        if self.use_per_channel_stddev_normalization and (self.per_channel_stddev is None or len(self.per_channel_stddev) != self.num_color_channels):
            self.per_channel_stddev = np.array(dataset_utils.calculate_per_channel_stddev(self.get_all_photos(), self.per_channel_mean, self.num_color_channels))
            self.logger.log('DataGenerator: Using per-channel stddev: {}'.format(list(self.per_channel_stddev)))

        # Use per-channel mean but in range [0, 255] if nothing else is given.
        # The normalization is done to the whole batch after transformations so
        # the images are not in range [-1,1] before transformations.
        if self.photo_cval is None:
            if self.per_channel_mean is None:
                self.photo_cval = np.array([0.0] * 3, dtype=np.float32)
            else:
                self.photo_cval = image_utils.np_from_normalized_to_255(self.per_channel_mean).astype(np.float32)
            self.logger.log('DataGenerator: Using photo cval: {}'.format(list(self.photo_cval)))

        # Use black (background)
        if self.mask_cval is None:
            self.mask_cval = np.array([0.0] * 3, dtype=np.float32)
            self.logger.log('DataGenerator: Using mask cval: {}'.format(list(self.mask_cval)))

        # Use the given random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    @abstractmethod
    def get_all_photos(self):
        # type: () -> list[ImageFile]

        """
        Returns all the photo ImageFile instances as a list

        # Arguments
            None
        # Returns
            :return: all the photo ImageFiles as a list
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    @abstractmethod
    def next(self):
        # type: () -> ()

        """
        This method should yield batches of data. The arguments for derived classes might vary.

        # Arguments
            None
        # Returns
            None
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    @abstractproperty
    def num_steps_per_epoch(self):
        # type: () -> int

        """
        This method returns the number of batches in epoch.

        # Arguments
            None
        # Returns
            :return: Number of batches (steps) per epoch
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    def _normalize_photo_batch(self, batch):
        # type: (np.array) -> np.array

        """
        Standardizes the color channels from the given image batch to zero-centered
        range [-1, 1] from the original [0, 255] range. In case a parameter is not supplied
        that normalization is not applied.

        # Arguments
            :param batch: numpy array with a batch of images to normalize
        # Returns
            :return: The parameter batch normalized with the given values
        """

        # Make sure the batch data type is correct
        batch = batch.astype(np.float32)

        if np.min(batch) < 0 or np.max(batch) > 255:
            raise ValueError('Batch image values are not between [0, 255], got [{}, {}]'.format(np.min(batch), np.max(batch)))

        # Map the values from [0, 255] to [-1, 1]
        batch = ((batch / 255.0) - 0.5) * 2.0

        # Subtract the per-channel-mean from the batch to "center" the data.
        if self.use_per_channel_mean_normalization:
            if self.per_channel_mean is None:
                raise ValueError('Use per-channel mean normalization is true, but per-channel mean is None')

            _per_channel_mean = np.array(self.per_channel_mean).astype(np.float32)

            # Per channel mean is in range [-1,1]
            if (_per_channel_mean >= -1.0 - 1e-7).all() and (_per_channel_mean <= 1.0 + 1e-7).all():
                batch -= _per_channel_mean
            # Per channel mean is in range [0, 255]
            elif (_per_channel_mean >= 0.0).all() and (_per_channel_mean <= 255.0).all():
                batch -= image_utils.np_from_255_to_normalized(_per_channel_mean)
            else:
                raise ValueError('Per channel mean is in unknown range: {}'.format(_per_channel_mean))

        # Additionally, you ideally would like to divide by the sttdev of
        # that feature or pixel as well if you want to normalize each feature
        # value to a z-score.
        if self.use_per_channel_stddev_normalization:
            if self.per_channel_stddev is None:
                raise ValueError('Use per-channel stddev normalization is true, but per-channel stddev is None')

            _per_channel_stddev = np.array(self.per_channel_stddev).astype(np.float32)

            # Per channel stddev is in range [-1, 1]
            if (_per_channel_stddev >= -1.0 - 1e-7).all() and (_per_channel_stddev <= 1.0 + 1e-7).all():
                batch /= _per_channel_stddev
            # Per channel stddev is in range [0, 255]
            elif (_per_channel_stddev >= 0.0).all() and (_per_channel_stddev <= 255.0).all():
                batch /= image_utils.np_from_255_to_normalized(_per_channel_stddev)
            else:
                raise ValueError('Per-channel stddev is in unknown range: {}'.format(_per_channel_stddev))

        return batch

#######################################
# SEGMENTATION DATA GENERATOR
#######################################


class SegmentationDataGenerator(DataGenerator):

    def __init__(self,
                 labeled_data_set,
                 unlabeled_data_set,
                 num_labeled_per_batch,
                 num_unlabeled_per_batch,
                 batch_data_format,
                 params,
                 class_weights=None,
                 label_generation_function=None):
        # type: (LabeledImageDataSet, UnlabeledImageDataSet, int, int, BatchDataFormat, DataGeneratorParameters, np.array[np.float32], Callable) -> None

        """
        # Arguments
            :param labeled_data_set: LabeledImageSet instance of the labeled data set
            :param unlabeled_data_set: UnlabeledImageSet instance of the unlabeled data set
            :param num_labeled_per_batch: number of labeled images per batch
            :param num_unlabeled_per_batch: number of unlabeled images per batch
            :param batch_data_format: format of the data batches
            :param params: DataGeneratorParameters object
            :param class_weights: class weights
            :param label_generation_function: function for label generation for unlabeled data
        """

        self.labeled_data_set = labeled_data_set
        self.unlabeled_data_set = unlabeled_data_set

        self.num_labeled_per_batch = num_labeled_per_batch
        self.num_unlabeled_per_batch = num_unlabeled_per_batch

        super(SegmentationDataGenerator, self).__init__(batch_data_format, params)

        if self.use_material_samples:
            if self.labeled_data_set.material_samples is None or len(self.labeled_data_set.material_samples) == 0:
                raise ValueError('Use material samples is true, but labeled data set does not contain material samples')

            self.labeled_data_iterator = MaterialSampleDataSetIterator(
                material_samples=self.labeled_data_set.material_samples,
                batch_size=num_labeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed,
                logger=self.logger,
                iter_mode=MaterialSampleIterationMode.UNIFORM_MEAN)

        else:
            self.labeled_data_iterator = BasicDataSetIterator(
                n=self.labeled_data_set.size,
                batch_size=num_labeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed,
                logger=self.logger)

        self.unlabeled_data_iterator = None

        if unlabeled_data_set is not None and unlabeled_data_set.size > 0 and num_unlabeled_per_batch > 0:
            self.unlabeled_data_iterator = BasicDataSetIterator(
                n=self.unlabeled_data_set.size,
                batch_size=num_unlabeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed,
                logger=self.logger)

        if labeled_data_set is None:
            raise ValueError('SegmentationDataGenerator does not support empty labeled data set')

        if class_weights is None:
            raise ValueError('Class weights is None. Use a numpy array of ones instead of None')

        self.class_weights = class_weights

        if label_generation_function is None:
            self.label_generation_function = SegmentationDataGenerator.default_label_generator_for_unlabeled_photos
        else:
            self.label_generation_function = label_generation_function

    def get_all_photos(self):
        # type: () -> list[ImageFile]

        """
        Returns all the photo file paths as a list

        # Arguments
            None
        # Returns
            :return: all the photo file paths as a list
        """

        photos = []
        photos += self.labeled_data_set.photo_image_set.image_files

        if self.using_unlabeled_data:
            photos += self.unlabeled_data_set.photo_image_set.image_files

        return photos

    @property
    def using_unlabeled_data(self):
        # type: () -> bool

        """
        Returns whether the generator is using unlabeled data.

        # Arguments
            None
        # Returns
            :return: true if there is unlabeled data otherwise false
        """
        return self.unlabeled_data_set is not None and \
               self.unlabeled_data_set.size > 0 and \
               self.unlabeled_data_iterator is not None and \
               self.num_unlabeled_per_batch > 0 and \
               self.batch_data_format != BatchDataFormat.SUPERVISED

    def get_labeled_batch_data(self, step_index, index_array):
        # type: (int, np.array[int]) -> (list[np.array], list[np.array])

        """
        # Arguments
            :param step_index: current step index
            :param index_array: indices of the labeled data
        # Returns
            :return: labeled data as three lists: (X, Y, WEIGHTS)
        """

        if self.use_material_samples:
            labeled_batch_files, material_samples = self.labeled_data_set.get_files_and_material_samples(index_array)
        else:
            labeled_batch_files, material_samples = self.labeled_data_set.get_indices(index_array), None

        # Process the labeled files for this batch
        labeled_data = [self.get_labeled_segmentation_data_pair(step_idx=step_index,
                                                                photo_file=labeled_batch_files[i][0],
                                                                mask_file=labeled_batch_files[i][1],
                                                                material_sample=material_samples[i] if material_samples is not None else None,
                                                                mask_type=SegmentationMaskEncodingType.INDEX) for i in range(len(labeled_batch_files))]

        # Unzip the photo mask pairs
        X, Y = zip(*labeled_data)

        # Create the weights for each ground truth segmentation
        W = []

        for y in Y:
            y_w = np.ones_like(y, dtype=np.float32)

            for i in range(len(self.class_weights)):
                mask = y[:, :] == i
                y_w[mask] = self.class_weights[i]

            W.append(y_w)

        return list(X), list(Y), W

    def get_unlabeled_batch_data(self, step_index, index_array):
        # type: (int, np.array[int]) -> (list[np.array], list[np.array], list[np.array])

        """
        # Arguments
            :param step_index: index of the current step
            :param index_array: indices of the unlabeled data
        # Returns
            :return: unlabeled data as three lists: (X, Y, WEIGHTS)
        """

        # If we don't have unlabeled data return two empty lists
        if not self.using_unlabeled_data:
            return [], [], []

        unlabeled_batch_files = self.unlabeled_data_set.get_indices(index_array)

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        unlabeled_data = [self.get_unlabeled_segmentation_data_pair(step_idx=step_index, photo_file=photo_file) for photo_file in unlabeled_batch_files]
        X_unlabeled, Y_unlabeled = zip(*unlabeled_data)
        W_unlabeled = []

        for y in Y_unlabeled:
            W_unlabeled.append(np.ones_like(y, dtype=np.float32))

        return list(X_unlabeled), list(Y_unlabeled), W_unlabeled

    def get_labeled_segmentation_data_pair(self, step_idx, photo_file, mask_file, material_sample, mask_type=SegmentationMaskEncodingType.INDEX):
        # type: (int, ImageFile, ImageFile, MaterialSample, SegmentationMaskEncodingType) -> (np.array, np.array)

        """
        Returns a photo mask pair for supervised segmentation training. Will apply data augmentation
        and cropping as instructed in the parameters.

        The photos are not normalized to range [-1,1] within the function.

        # Arguments
            :param step_idx: index of the current training step
            :param photo_file: photo as ImageFile
            :param mask_file: segmentation mask as ImageFile
            :param material_sample: material sample information for the files
            :param mask_type:
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        # Load the image and mask as PIL images
        photo = photo_file.get_image(color_channels=self.num_color_channels)
        mask = mask_file.get_image(color_channels=3)

        # Resize the photo to match the mask size if necessary, since
        # the original photos are sometimes huge
        if photo.size != mask.size:
            photo = photo.resize(mask.size, Image.ANTIALIAS)

        if photo.size != mask.size:
            raise ValueError('Non-matching photo and mask dimensions after resize: {} != {}'.format(photo.size, mask.size))

        # Convert to numpy array
        np_photo = img_to_array(photo)
        np_mask = img_to_array(mask)

        # Apply crops and augmentation
        np_photo, np_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx, np_photo=np_photo, np_mask=np_mask, photo_cval=self.photo_cval, mask_cval=self.mask_cval, material_sample=material_sample, retry_crops=True)

        # Expand the mask image to the one-hot encoded shape: H x W x NUM_CLASSES
        if mask_type == SegmentationMaskEncodingType.ONE_HOT:
            np_mask = dataset_utils.one_hot_encode_mask(np_mask, self.material_class_information)

            # Sanity check: material samples are supposed to guarantee material instances
            # One-hot encoded mask
            if material_sample is not None:
                if not np.any(np.not_equal(np_mask[:, :, material_sample.material_id], 0)):
                    self.logger.log_image(np_photo, file_name='{}_crop_missing_{}.jpg'.format(photo_file.file_name, material_sample.material_id))
                    self.logger.warn('Material sample for material id {} was given but no corresponding entries were found in the cropped mask. Found: {}'
                                     .format(material_sample.material_id, list(np.unique(np.argmax(np_mask)))))
        elif mask_type == SegmentationMaskEncodingType.INDEX:
            np_mask = dataset_utils.index_encode_mask(np_mask, self.material_class_information)

            # Sanity check: material samples are supposed to guarantee material instances
            # Index encoded mask
            if material_sample is not None:
                if not np.any(np.equal(np_mask, material_sample.material_id)):
                    self.logger.log_image(np_photo, file_name='{}_crop_missing_{}.jpg'.format(photo_file.file_name, material_sample.material_id))
                    self.logger.warn('Material sample for material id {} was given but no corresponding entries were found in the cropped mask. Found: {}'
                                     .format(material_sample.material_id, list(np.unique(np_mask))))
        else:
            raise ValueError('Unknown mask_type: {}'.format(mask_type))

        return np_photo, np_mask

    def get_unlabeled_segmentation_data_pair(self, step_idx, photo_file):
        # type: (int, ImageFile) -> (np.array, np.array)

        """
        Returns a photo mask pair for semi-supervised/unsupervised segmentation training.
        Will apply data augmentation and cropping as instructed in the parameters.

        The photos are not normalized to range [-1,1] within the function.

        # Arguments
            :param step_idx: index of the current training step
            :param photo_file: an ImageFile of the photo
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        # Load the photo as PIL image
        photo_file = photo_file.get_image(color_channels=self.num_color_channels)
        np_photo = img_to_array(photo_file)

        # Generate mask for the photo - note: the labels are generated before cropping
        # and augmentation to capture global structure within the image
        np_mask = self.label_generation_function(np_photo)

        # Expand the last dimension of the mask to make it compatible with augmentation functions
        np_mask = np_mask[:, :, np.newaxis]

        # Apply crops and augmentation
        np_photo, np_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx, np_photo=np_photo, np_mask=np_mask, photo_cval=self.photo_cval, mask_cval=[0], retry_crops=False)

        # Squeeze the unnecessary last dimension out
        np_mask = np.squeeze(np_mask)

        # Map the mask values back to a continuous range [0, N_SUPERPIXELS]. The values
        # might be non-continuous due to cropping and augmentation
        old_indices = np.sort(np.unique(np_mask))
        new_indices = np.arange(np.max(old_indices + 1))
        num_indices = len(old_indices)

        for i in range(0, num_indices):
            # If the indices match - do nothing
            if old_indices[i] == new_indices[i]:
                continue

            index_mask = np_mask[:, :] == old_indices[i]
            np_mask[index_mask] = new_indices[i]

        return np_photo, np_mask

    def process_segmentation_photo_mask_pair(self, step_idx, np_photo, np_mask, photo_cval, mask_cval, material_sample=None, retry_crops=True):
        # type: (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MaterialSample, bool) -> (np.ndarray, np.ndarray)

        """
        Applies crop and data augmentation to two numpy arrays representing the photo and
        the respective segmentation mask. The photos are not normalized to range [-1,1]
        within the function.

        # Arguments
            :param step_idx: index of the current step
            :param np_photo: the photo as a numpy array
            :param np_mask: the mask as a numpy array must have same spatial dimensions (HxW) as np_photo
            :param photo_cval: photo fill value in range [0, 255]
            :param mask_cval: mask fill value in range [0, 255]
            :param material_sample: the material sample
            :param retry_crops: retries crops if the whole crop is 0 (BG)
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        if np_photo.shape[:2] != np_mask.shape[:2]:
            raise ValueError('Non-matching photo and mask shapes: {} != {}'.format(np_photo.shape, np_mask.shape))

        if material_sample is not None and self.crop_shape is None:
            raise ValueError('Cannot use material samples without cropping')

        # Check whether we need to resize the photo and the mask to a constant size
        if self.resize_shape is not None:
            np_photo = image_utils.np_scale_image_with_padding(np_photo, shape=self.resize_shape, cval=photo_cval, interp='bilinear')
            np_mask = image_utils.np_scale_image_with_padding(np_mask, shape=self.resize_shape, cval=mask_cval, interp='nearest')

        # Check whether any of the image dimensions is smaller than the crop,
        # if so pad with the assigned fill colors
        if self.crop_shape is not None and (np_photo.shape[0] < self.crop_shape[0] or np_photo.shape[1] < self.crop_shape[1]):
            # Image dimensions must be at minimum the same as the crop dimension
            # on each axis. The photo needs to be filled with the photo_cval color and masks
            # with the mask cval color
            min_img_shape = (max(self.crop_shape[0], np_photo.shape[0]), max(self.crop_shape[1], np_photo.shape[1]))
            np_photo = image_utils.np_pad_image_to_shape(np_photo, min_img_shape, photo_cval)
            np_mask = image_utils.np_pad_image_to_shape(np_mask, min_img_shape, mask_cval)

        # If we are using data augmentation apply the random transformation
        # to both the image and mask now. We apply the transformation to the
        # whole image to decrease the number of 'dead' pixels due to transformations
        # within the possible crop.
        bbox = material_sample.get_bbox_abs() if material_sample is not None else None

        if self.use_data_augmentation and np.random.random() <= self.data_augmentation_params.augmentation_probability_function(step_idx):

            orig_vals, np_orig_photo, np_orig_mask = None, None, None

            if material_sample is not None:
                np_orig_photo = np.array(np_photo, copy=True)
                np_orig_mask = np.array(np_mask, copy=True)

            images, transform = image_utils.np_apply_random_transform(images=[np_photo, np_mask],
                                                                      cvals=[photo_cval, mask_cval],
                                                                      fill_mode=self.data_augmentation_params.fill_mode,
                                                                      interpolations=[ImageInterpolation.BICUBIC, ImageInterpolation.NEAREST],
                                                                      img_data_format=self.img_data_format,
                                                                      rotation_range=self.data_augmentation_params.rotation_range,
                                                                      zoom_range=self.data_augmentation_params.zoom_range,
                                                                      width_shift_range=self.data_augmentation_params.width_shift_range,
                                                                      height_shift_range=self.data_augmentation_params.height_shift_range,
                                                                      channel_shift_ranges=[self.data_augmentation_params.channel_shift_range, None],
                                                                      horizontal_flip=self.data_augmentation_params.horizontal_flip,
                                                                      vertical_flip=self.data_augmentation_params.vertical_flip,
                                                                      gamma_adjust_ranges=[self.data_augmentation_params.gamma_adjust_range, None])
            # Unpack the images
            np_photo, np_mask = images

            if material_sample is not None:
                bbox = self.transform_bbox(bbox, transform)

                # If we could not decode from the augmented photo, skip the augmentation and
                # default to the original bbox from the material sample
                if bbox is None:
                    bbox = material_sample.get_bbox_abs()
                    np_photo = np_orig_photo
                    np_mask = np_orig_mask
                    self.logger.debug_log('Could not recover a valid bbox after augmentation - reverting to original input and material sample')
                if not self.bbox_contains_material_sample(np_mask, bbox, material_sample):
                    bbox = material_sample.get_bbox_abs()
                    np_photo = np_orig_photo
                    np_mask = np_orig_mask
                    self.logger.warn('Bbox did not contain material sample after augmentation - reverting to original input and material sample')

        # If a crop size is given: take a random crop of both the image and the mask
        if self.crop_shape is not None:
            if dataset_utils.count_trailing_zeroes(self.crop_shape[0]) < self.div2_constraint or \
                            dataset_utils.count_trailing_zeroes(self.crop_shape[1]) < self.div2_constraint:
                raise ValueError('The crop size does not satisfy the div2 constraint of {}'.format(self.div2_constraint))

            # If we don't have a bounding box as a hint for cropping - take random crops
            if bbox is None:
                # Re-attempt crops if the crops end up getting only background
                # i.e. black pixels
                attempts = 5

                while attempts > 0:
                    x1y1, x2y2 = image_utils.np_get_random_crop_area(np_mask, self.crop_shape[1], self.crop_shape[0])
                    mask_crop = image_utils.np_crop_image(np_mask, x1y1[0], x1y1[1], x2y2[0], x2y2[1])

                    # If the mask crop is only background (all R channel is zero) - try another crop
                    # until attempts are exhausted
                    if np.max(mask_crop[:, :, 0]) == 0 and attempts - 1 != 0 and retry_crops:
                        attempts -= 1
                        continue

                    np_mask = mask_crop
                    np_photo = image_utils.np_crop_image(np_photo, x1y1[0], x1y1[1], x2y2[0], x2y2[1])
                    break
            # Use the bounding box information to take a targeted crop
            else:
                tlc, trc, brc, blc = bbox
                crop_height = self.crop_shape[0]
                crop_width = self.crop_shape[1]
                bbox_ymin = tlc[0]
                bbox_ymax = blc[0]
                bbox_xmin = tlc[1]
                bbox_xmax = trc[1]
                bbox_height = bbox_ymax - bbox_ymin
                bbox_width = bbox_xmax - bbox_xmin
                height_diff = abs(bbox_height - crop_height)
                width_diff = abs(bbox_width - crop_width)

                self.logger.debug_log('Bbox width: {}, height: {}'.format(bbox_width, bbox_height))

                # If the crop can fit the whole material sample within it
                if bbox_height < crop_height and bbox_width < crop_width:
                    crop_ymin = bbox_ymin - np.random.randint(0, min(height_diff, bbox_ymin + 1))
                    crop_ymax = crop_ymin + crop_height
                    crop_xmin = bbox_xmin - np.random.randint(0, min(width_diff, bbox_xmin + 1))
                    crop_xmax = crop_xmin + crop_width

                # If the crop can't fit the whole material sample within it
                else:
                    crop_ymin = bbox_ymin + np.random.randint(0, height_diff + 1)
                    crop_ymax = crop_ymin + crop_height
                    crop_xmin = bbox_xmin + np.random.randint(0, width_diff + 1)
                    crop_xmax = crop_xmin + crop_width

                # Sanity check for y values
                if crop_ymax > np_mask.shape[0]:
                    diff = crop_ymax - np_mask.shape[0]
                    crop_ymin = crop_ymin - diff
                    crop_ymax = crop_ymax - diff

                # Sanity check for x values
                if crop_xmax > np_mask.shape[1]:
                    diff = crop_xmax - np_mask.shape[1]
                    crop_xmin = crop_xmin - diff
                    crop_xmax = crop_xmax - diff

                np_mask = image_utils.np_crop_image(np_mask, crop_xmin, crop_ymin, crop_xmax, crop_ymax)
                np_photo = image_utils.np_crop_image(np_photo, crop_xmin, crop_ymin, crop_xmax, crop_ymax)

        # If a crop size is not given, make sure the image dimensions satisfy
        # the div2_constraint i.e. are n times divisible by 2 to work within
        # the network. If the dimensions are not ok pad the images.
        img_height_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[0])
        img_width_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[1])

        if img_height_div2 < self.div2_constraint or img_width_div2 < self.div2_constraint:
            # The photo needs to be filled with the photo_cval color and masks with the mask cval color
            padded_shape = dataset_utils.get_required_image_dimensions(np_photo.shape, self.div2_constraint)
            np_photo = image_utils.np_pad_image_to_shape(np_photo, padded_shape, photo_cval)
            np_mask = image_utils.np_pad_image_to_shape(np_mask, padded_shape, mask_cval)

        return np_photo, np_mask

    def transform_bbox(self, bbox, transform):
        # type: (tuple[tuple[int, int]], ImageTransform) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]

        # Transform the bbox coords to ndarray
        # Bbox is: tlc, trc, brc, blc
        bbox_coords = np.array(bbox, dtype=np.float32)

        # Sanity check the dimensions
        if bbox_coords.shape[0] != 4 and bbox_coords.shape[1] != 2:
            raise ValueError('Expected bounding box with dimensions [4,2] got shape: {}'.format(bbox_coords.shape))

        # The transform needs to get the coordinates in xy instead of yx, so flip the coordinates
        bbox_coords = np.fliplr(bbox_coords)

        # Transform the bounding box corner coordinates - cast to int32
        corners = np.round(transform.transform_coordinates(bbox_coords)).astype(dtype=np.int32)

        # Figure out which corners have gone out of image boundaries
        out_of_bounds_corner_indices = [i for i in range(len(corners)) if not (0 <= corners[i][0] <= transform.image_width and 0 <= corners[i][1] <= transform.image_height)]
        corners = [corner for i, corner in enumerate(corners) if i not in out_of_bounds_corner_indices]

        # If we have less than two corners we cannot rebuild the bounding box
        if len(out_of_bounds_corner_indices) > 2 or len(corners) < 2:
            return None
        # tlc and brc out of bounds
        if 0 in out_of_bounds_corner_indices and 2 in out_of_bounds_corner_indices:
            return None
        # trc and blc out of bounds
        if 1 in out_of_bounds_corner_indices and 3 in out_of_bounds_corner_indices:
            return None

        # If we have two corners that were originally opposite corners we can rebuild
        # the axis-aligned bounding box
        x_coords, y_coords = zip(*corners)

        y_min = min(y_coords)
        y_max = max(y_coords)
        x_min = min(x_coords)
        x_max = max(x_coords)

        y_diff = y_max - y_min
        x_diff = x_max - x_min

        # It is possible that the interpolation has produced additional
        # corner color values. Check that the difference is at least three
        # pixels between the minimum and maximum values.
        if y_diff <= 3 or x_diff <= 3:
            return None

        # Rebuild the bounding box and represent in (y,x)
        tlc = (y_min, x_min)
        trc = (y_min, x_max)
        brc = (y_max, x_max)
        blc = (y_max, x_min)

        return tlc, trc, brc, blc

    def bbox_contains_material_sample(self, np_img, bbox, material_sample):
        # type: (np.ndarray, tuple[tuple[int, int]], MaterialSample) -> bool

        # Transform the bbox coords to ndarray
        # Bbox is: tlc, trc, brc, blc
        bbox_coords = np.array(bbox, dtype=np.int32)

        # Sanity check the dimensions
        if bbox_coords.shape[0] != 4 and bbox_coords.shape[1] != 2:
            raise ValueError('Expected bounding box with dimensions [4,2] got shape: {}'.format(bbox_coords.shape))

        if np_img.ndim != 3:
            raise ValueError('Expected a segmentation mask image with shape [H,W,C] got ndim: {}'.format(np_img.ndim))

        y_min = bbox[0][0]
        y_max = bbox[2][0]
        x_min = bbox[0][1]
        x_max = bbox[1][1]

        # One-hot encoded: select the red channel from the bounding box area
        np_bbox_img_area = np_img[y_min:y_max, x_min:x_max, 0]

        if np.any(np.equal(np_bbox_img_area, material_sample.material_r_color)):
            return True

        return False

    def next(self):
        # type: (int, int, tuple[int]) -> (list[np.ndarray], list[np.ndarray])

        """
        Generates batches of data for semi supervised mean teacher training.

        # Arguments
            None
        # Returns
            :return: A batch of data for semi supervised training as a tuple of (X,Y).
            The input data (X) has the following:

            0: Input images consisting of both labeled and unlabeled images. The unlabeled images are
               the M last images in the batch.
            1: Labels of the input images. This set has labels for both labeled and unlabeled images
               and the unlabeled images.
            2: Number of unlabeled data in the X and Y (photos and labels).
        """

        with self.lock:
            labeled_index_array, labeled_current_index, labeled_current_batch_size, labeled_step_index = self.labeled_data_iterator.get_next_batch()
            unlabeled_index_array, unlabeled_current_index, unlabeled_current_batch_size, unlabeled_step_index = None, 0, 0, 0

            if self.using_unlabeled_data:
                unlabeled_index_array, unlabeled_current_index, unlabeled_current_batch_size, unlabeled_step_index = self.unlabeled_data_iterator.get_next_batch()

        X, Y, W = self.get_labeled_batch_data(labeled_step_index, labeled_index_array)
        X_unlabeled, Y_unlabeled, W_unlabeled = self.get_unlabeled_batch_data(unlabeled_step_index, unlabeled_index_array)
        X = X + X_unlabeled
        Y = Y + Y_unlabeled
        W = W + W_unlabeled

        num_unlabeled_samples_in_batch = len(X_unlabeled)
        num_samples_in_batch = len(X)

        # Cast the lists to numpy arrays
        X, Y, W = np.array(X, dtype=np.float32, copy=False), np.array(Y, dtype=np.int32, copy=False), np.array(W, dtype=np.float32, copy=False)

        # Normalize the photo batch data
        X = self._normalize_photo_batch(X)

        if self.batch_data_format == BatchDataFormat.SUPERVISED:
            batch_input_data = [X]
            batch_output_data = [np.expand_dims(Y, -1)]
        elif self.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # The dimensions of the number of unlabeled in the batch must match with batch dimension
            num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

            # Generate a dummy output for the dummy loss function and yield a batch of data
            dummy_output = np.zeros(shape=[num_samples_in_batch], dtype=np.int32)

            batch_input_data = [X, Y, W, num_unlabeled]

            if X.shape[0] != Y.shape[0] or X.shape[0] != W.shape[0] or X.shape[0] != num_unlabeled.shape[0]:
                self.logger.warn('Unmatching input first (batch) dimensions: {}, {}, {}, {}'.format(X.shape[0], Y.shape[0], W.shape[0], num_unlabeled.shape[0]))

            # Provide the true classification masks for labeled samples only - these go to the second loss function
            # in the semi-supervised model that is only used to calculate metrics. The output has to have the same
            # rank as the output from the network
            logits_output = np.expand_dims(np.copy(Y), -1)
            logits_output[num_samples_in_batch-num_unlabeled_samples_in_batch:] = 0
            batch_output_data = [dummy_output, logits_output]
        else:
            raise ValueError('Unknown batch data format: {}'.format(self.batch_data_format))

        return batch_input_data, batch_output_data

    @property
    def num_steps_per_epoch(self):
        # type: () -> int

        """
        Returns the number of batches (steps) per epoch.

        # Arguments
            None
        # Returns
            :return: Number of steps per epoch
        """
        return self.labeled_data_iterator.num_steps_per_epoch

    @staticmethod
    def default_label_generator_for_unlabeled_photos(np_image):
        # type: (np.ndarray) -> np.ndarray

        """
        Default label generator function for unlabeled photos. The label generator should
        encode the information in the form HxW.

        # Arguments
            :param np_image: the image as a numpy array
        # Returns
            :return: numpy array of same shape as np_image with every labeled as 0
        """
        return np.zeros(shape=(np_image.shape[0], np_image.shape[1]), dtype=np.int32)


################################################
# CLASSIFICATION DATA GENERATOR
################################################


class MINCClassificationDataGenerator(DataGenerator):

    def __init__(self,
                 minc_data_set_file_path,
                 minc_labels_translation_file_path,
                 minc_photos_folder_path,
                 num_labeled_per_batch,
                 params):
        # type: (str, str, str, int, DataGeneratorParameters) -> None

        print 'Loading MINC data set from file: {}'.format(minc_data_set_file_path)
        self.minc_data_set_file_path = minc_data_set_file_path
        self.data_set = self._read_minc_data_set_file(minc_data_set_file_path)
        print 'Loaded MINC data set with {} samples'.format(len(self.data_set))

        print 'Reading MINC labels translation file: {}'.format(minc_labels_translation_file_path)
        self.minc_labels_translation_file_path = minc_data_set_file_path
        self.minc_to_custom_label_mapping = self._read_minc_labels_translation_file(minc_labels_translation_file_path)
        self.num_labels = len(self.minc_to_custom_label_mapping)
        print 'Loaded {} label mappings'.format(self.num_labels)

        print 'Reading MINC photos to ImageSet from: {}'.format(minc_photos_folder_path)
        self.minc_photos_folder_path = minc_photos_folder_path
        self.minc_photos_image_set = ImageSet('photos', minc_photos_folder_path)
        print 'Constructed ImageSet with {} images'.format(self.minc_photos_image_set.size)

        if self.minc_photos_image_set is None or self.minc_photos_image_set.size <= 0:
            raise ValueError('Could not find MINC photos from: {}'.format(self.minc_photos_folder_path))

        super(MINCClassificationDataGenerator, self).__init__(params)

        self.data_iterator = BasicDataSetIterator(
            n=len(self.data_set),
            batch_size=num_labeled_per_batch,
            shuffle=self.shuffle_data_after_epoch,
            seed=self.random_seed)

    def get_all_photos(self):
        # type: () -> list[ImageFile]

        """
        Returns all the photo ImageFile instances as a list

        # Arguments
            None
        # Returns
            :return: all the photo ImageFiles as a list
        """
        return self.minc_photos_image_set.image_files

    @property
    def num_steps_per_epoch(self):
        return self.data_iterator.num_steps_per_epoch

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size, step_index = self.data_iterator.get_next_batch()

        # Get the samples
        data = [self._get_sample(step_index, self.data_set[sample_idx]) for sample_idx in index_array]
        X, Y = zip(*data)

        # Debug: Write images
        if settings.DEBUG:
            for i in range(len(X)):
                self.logger.save_debug_image(X[i], file_name='{}_{}_{}_photo.jpg'.format(step_index, i, np.argmax(Y[i])), format='JPEG')
        # End of: debug

        X = np.array(X)
        Y = np.array(Y)

        # Normalize the photo batch data
        X = self._normalize_photo_batch(X)

        return [X], Y

    def _get_sample(self, step_idx, minc_sample):
        # type: (int, MINCSample) -> (np.ndarray[np.float32], np.ndarray[np.float32])

        # Read the photo file from the photos ImageSet
        image_file = self.minc_photos_image_set.get_image_file_by_file_name(minc_sample.photo_id)

        if image_file is None:
            raise ValueError('Could not find image from ImageSet with file name: {}'.format(minc_sample.photo_id))

        if self.crop_shape is None:
            raise ValueError('MINCClassificationDataGenerator cannot be used without setting a crop shape')

        np_photo = img_to_array(image_file.get_image(self.num_color_channels))
        image_height, image_width = np_photo.shape[0], np_photo.shape[1]
        crop_height, crop_width = self.crop_shape[0], self.crop_shape[1]
        crop_center_y = minc_sample.y
        crop_center_x = minc_sample.x

        # Apply data augmentation
        if self.use_data_augmentation and np.random.random() <= self.data_augmentation_params.augmentation_probability_function(step_idx):
            np_photo_orig = np.array(np_photo, copy=True)
            images, transform = image_utils.np_apply_random_transform(images=[np_photo],
                                                                      cvals=[self.photo_cval],
                                                                      fill_mode=self.data_augmentation_params.fill_mode,
                                                                      interpolations=[ImageInterpolation.BICUBIC],
                                                                      transform_origin=np.array([crop_center_y, crop_center_x]),
                                                                      img_data_format=self.img_data_format,
                                                                      rotation_range=self.data_augmentation_params.rotation_range,
                                                                      zoom_range=self.data_augmentation_params.zoom_range,
                                                                      width_shift_range=self.data_augmentation_params.width_shift_range,
                                                                      height_shift_range=self.data_augmentation_params.height_shift_range,
                                                                      channel_shift_ranges=[self.data_augmentation_params.channel_shift_range],
                                                                      horizontal_flip=self.data_augmentation_params.horizontal_flip,
                                                                      vertical_flip=self.data_augmentation_params.vertical_flip,
                                                                      gamma_adjust_ranges=[self.data_augmentation_params.gamma_adjust_range])

            # Unpack the photo
            np_photo, = images

            crop_center = transform.transform_normalized_coordinates(np.array([minc_sample.x, minc_sample.y]))
            crop_center_x_new, crop_center_y_new = crop_center[0], crop_center[1]

            # If the center has gone out of bounds abandon the augmentation - otherwise, update the crop center values
            if (not 0.0 < crop_center_y_new < 1.0) or (not 0.0 < crop_center_x_new < 1.0):
                np_photo = np_photo_orig
            else:
                crop_center_y = crop_center_y_new
                crop_center_x = crop_center_x_new

        # Crop the image with the specified crop center. Regions going out of bounds are padded with a
        # constant value.
        y_c = crop_center_y*image_height
        x_c = crop_center_x*image_width
        y_0 = int(round(y_c - crop_height*0.5))
        x_0 = int(round(x_c - crop_width*0.5))
        y_1 = int(round(y_c + crop_height*0.5))
        x_1 = int(round(x_c + crop_width*0.5))

        np_photo = image_utils.np_crop_image_with_fill(np_photo, x1=x_0, y1=y_0, x2=x_1, y2=y_1, cval=self.photo_cval)

        # Construct the one-hot vector
        custom_label = self.minc_to_custom_label_mapping[minc_sample.minc_label]
        y = np.zeros(self.num_labels)
        y[custom_label] = 1.0

        return np_photo, y

    def _read_minc_data_set_file(self, file_path):
        data_set = list()

        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                # Each line of the file is: 4-tuple list of (label, photo id, x, y)
                label, photo_id, x, y = line.split(',')
                data_set.append(MINCSample(label=int(label), photo_id=photo_id.strip(), x=float(x), y=float(y)))

        return data_set

    def _read_minc_labels_translation_file(self, file_path):
        minc_to_custom_label_mapping = dict()

        with open(file_path, 'r') as f:
            # The first line should be skipped because it describes the data, which is
            # in the format of substance_name,minc_class_idx,custom_class_idx
            lines = f.readlines()

            for idx, line in enumerate(lines):
                # Skip the first line
                if idx == 0:
                    continue

                substance_name, minc_class_idx, custom_class_idx = line.split(',')

                # Check that there are no duplicate entries for MINC class ids
                if minc_to_custom_label_mapping.has_key(int(minc_class_idx)):
                    raise ValueError('Label mapping already contains entry for MINC class id: {}'.format(int(minc_class_idx)))

                # Check that there are no duplicate entries for custom class ids
                if int(custom_class_idx) in minc_to_custom_label_mapping.values():
                    raise ValueError('Label mapping already contains entry for custom class id: {}'.format(int(custom_class_idx)))

                minc_to_custom_label_mapping[int(minc_class_idx)] = int(custom_class_idx)

        return minc_to_custom_label_mapping
