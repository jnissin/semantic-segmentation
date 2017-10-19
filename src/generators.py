# coding=utf-8

import time
import random
import numpy as np

from enum import Enum
from abc import ABCMeta, abstractmethod, abstractproperty

from PIL import Image

from utils import dataset_utils
from utils import image_utils
from utils.image_utils import ImageInterpolation, ImageTransform, img_to_array
from utils.dataset_utils import MaterialClassInformation, MaterialSample, MINCSample
from data_set import LabeledImageDataSet, UnlabeledImageDataSet, ImageFile, ImageSet
from iterators import DataSetIterator, BasicDataSetIterator, MaterialSampleDataSetIterator, MaterialSampleIterationMode
from logger import Logger
from enums import BatchDataFormat, SuperpixelSegmentationFunctionType

from scipy.ndimage.interpolation import shift

from joblib import Parallel, delayed

import settings


_UUID_COUNTER = 0


def pickle_method(instance, name, *args, **kwargs):
    "indirect caller for instance methods and multiprocessing"
    if kwargs is None:
        kwargs = {}
    return getattr(instance, name)(*args, **kwargs)


def _get_next_uuid():
    # type: () -> int
    global _UUID_COUNTER
    uuid = _UUID_COUNTER
    _UUID_COUNTER += 1
    return uuid


#######################################
# UTILITY CLASSES
#######################################

class DataAugmentationParameters:
    """
    This class helps to maintain the data augmentation parameters for data generators.
    """

    """
    This static data store helps make unpickable parameters such as ramp-up
    functions pickable
    """

    DATA_STORE = {}

    @staticmethod
    def _get_ds_item(key):
        return DataAugmentationParameters.DATA_STORE.get(key, None)

    @staticmethod
    def _set_ds_item(key, value):
        DataAugmentationParameters.DATA_STORE[key] = value

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
                 gamma_adjust_range=0.0,
                 mean_teacher_noise_params=None):
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
            expected to be in range [0,1],
            :param mean_teacher_noise_params: Parameters for the mean teacher noise generation
        # Returns
            :return: A new instance of DataAugmentationParameters
        """

        if augmentation_probability_function is None:
            raise ValueError('Augmentation probability function cannot be None - needs to be an evaluatable lambda function string')

        self._augmentation_probability_function_str = augmentation_probability_function
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self._gaussian_noise_stddev_function_str = gaussian_noise_stddev_function
        self.mean_teacher_noise_params = mean_teacher_noise_params

        if self._augmentation_probability_function_str is not None:
            DataAugmentationParameters._set_ds_item(self._augmentation_probability_function_str, eval(self._augmentation_probability_function_str))

        if self._gaussian_noise_stddev_function_str is not None:
            DataAugmentationParameters._set_ds_item(self._gaussian_noise_stddev_function_str, eval(self._gaussian_noise_stddev_function_str))

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

    @property
    def augmentation_probability_function(self):
        if self._augmentation_probability_function_str is not None:
            return DataAugmentationParameters._get_ds_item(self._augmentation_probability_function_str)

        return None

    @property
    def gaussian_noise_stddev_function(self):
        if self._gaussian_noise_stddev_function_str is not None:
            return DataAugmentationParameters._get_ds_item(self._gaussian_noise_stddev_function_str)

        return None

    @property
    def using_gaussian_noise(self):
        return self.gaussian_noise_stddev_function is not None

    @property
    def using_mean_teacher_noise(self):
        return self.mean_teacher_noise_params is not None


class DataGeneratorParameters(object):
    """
    This class helps to maintain parameters in common with all different data generators.
    """

    def __init__(self,
                 num_color_channels,
                 name="",
                 random_seed=None,
                 crop_shapes=None,
                 resize_shapes=None,
                 use_per_channel_mean_normalization=True,
                 per_channel_mean=None,
                 use_per_channel_stddev_normalization=True,
                 per_channel_stddev=None,
                 photo_cval=None,
                 use_data_augmentation=False,
                 data_augmentation_params=None,
                 shuffle_data_after_epoch=True,
                 div2_constraint=0,
                 initial_epoch=0,
                 generate_mean_teacher_data=False):
        """
        Builds a wrapper for DataGenerator parameters

        # Arguments
            :param num_color_channels: number of channels in the photos; 1, 3 or 4
            :param name: name for the generator so it can be recognized from e.g. logs
            :param random_seed: an integer random seed
            :param crop_shapes: size of the crop or None if no cropping should be applied
            :param resize_shapes: size of the desired resized images, None if no resizing should be applied
            :param use_per_channel_mean_normalization: whether per-channel mean normalization should be applied
            :param per_channel_mean: per channel mean in range [-1, 1]
            :param use_per_channel_stddev_normalization: whether per-channel stddev normalizaiton should be applied
            :param per_channel_stddev: per-channel stddev in range [-1, 1]
            :param photo_cval: fill color value for photos [0,255], otherwise per-channel mean used
            :param use_data_augmentation: should data augmentation be used
            :param data_augmentation_params: parameters for data augmentation
            :param shuffle_data_after_epoch: should the data be shuffled after every epoch
            :param div2_constraint: how many times does the image/crop need to be divisible by two
            :param initial_epoch: number of the initial epoch
            :param generate_mean_teacher_data: should we generate mean teacher data? If so it is appended as last item of the input data
        # Returns
            Nothing
        """

        self.num_color_channels = num_color_channels
        self.name = name
        self.random_seed = random_seed
        self.crop_shapes = crop_shapes
        self.resize_shapes = resize_shapes
        self.use_per_channel_mean_normalization = use_per_channel_mean_normalization
        self.per_channel_mean = per_channel_mean
        self.use_per_channel_stddev_normalization = use_per_channel_stddev_normalization
        self.per_channel_stddev = per_channel_stddev
        self.photo_cval = photo_cval
        self.use_data_augmentation = use_data_augmentation
        self.data_augmentation_params = data_augmentation_params
        self.shuffle_data_after_epoch = shuffle_data_after_epoch
        self.div2_constraint = div2_constraint
        self.initial_epoch = initial_epoch
        self.generate_mean_teacher_data = generate_mean_teacher_data


class SegmentationDataGeneratorParameters(DataGeneratorParameters):
    """
    Helps to maintain parameters of the segmentation data generator
    """

    def __init__(self,
                 material_class_information,
                 mask_cval=None,
                 use_material_samples=False,
                 use_selective_attention=False,
                 use_adaptive_sampling=False,
                 num_crop_reattempts=0,
                 **kwargs):
        # type: (list[MaterialClassInformation], np.ndarray, bool, bool, bool, int) -> None

        """
        Builds a wrapper for SegmentationDataGenerator parameters

        # Arguments
            :param material_class_information: material class information list
            :param mask_cval: fill color value for masks [0,255], otherwise zeros matching mask encoding are used
            :param use_material_samples: should material samples be used
            :param use_selective_attention: should we use selective attention (mark everything else as bg besides the material sample material)
            :param use_adaptive_sampling: should we use adaptive sampling (adapt sampling probability according to pixels seen per category)
        # Returns
            Nothing
        """
        super(SegmentationDataGeneratorParameters, self).__init__(**kwargs)

        self.material_class_information = material_class_information
        self.mask_cval = mask_cval
        self.use_material_samples = use_material_samples
        self.use_selective_attention = use_selective_attention
        self.use_adaptive_sampling = use_adaptive_sampling
        self.num_crop_reattempts = num_crop_reattempts


#######################################
# DATA GENERATOR
#######################################


class SegmentationMaskEncodingType(Enum):
    INDEX = 0
    ONE_HOT = 1


class DataGenerator(object):

    """
    Abstract class which declares necessary methods for different DataGenerators. Also,
    unwraps the DataGeneratorParameters to class member variables.

    The DataGenerator class should be pickable i.e. everything should work when the class
    is copied to multiple different processes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, batch_data_format, params):
        # type: (BatchDataFormat, DataGeneratorParameters) -> None

        # UUID might be before the init function is called
        if not hasattr(self, '_uuid'):
            self._uuid = None

        self._logger = None

        self.logger.log('DataGenerator: Initializing data generator with UUID: {}'.format(self.uuid))
        self.batch_data_format = batch_data_format

        # Unwrap DataGeneratorParameters to member variables
        self.num_color_channels = params.num_color_channels
        self.random_seed = params.random_seed
        self.name = params.name
        self._crop_shapes = params.crop_shapes
        self._resize_shapes = params.resize_shapes
        self.use_per_channel_mean_normalization = params.use_per_channel_mean_normalization
        self.per_channel_mean = params.per_channel_mean
        self.use_per_channel_stddev_normalization = params.use_per_channel_stddev_normalization
        self.per_channel_stddev = params.per_channel_stddev
        self.photo_cval = params.photo_cval
        self.use_data_augmentation = params.use_data_augmentation
        self.data_augmentation_params = params.data_augmentation_params
        self.shuffle_data_after_epoch = params.shuffle_data_after_epoch
        self.div2_constraint = params.div2_constraint
        self.initial_epoch = params.initial_epoch
        self.generate_mean_teacher_data = params.generate_mean_teacher_data

        # Other member variables
        self.img_data_format = settings.DEFAULT_IMAGE_DATA_FORMAT

        # Use the given random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

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

        self.using_random_crop_sizes = any(isinstance(i, list) for i in self._crop_shapes) if self._crop_shapes is not None else False
        self.logger.log('DataGenerator: Using random crop sizes: {}'.format(self.using_random_crop_sizes))
        self.using_random_resize_sizes = any(isinstance(i, list) for i in self._resize_shapes) if self._resize_shapes is not None else False
        self.logger.log('DataGenerator: Using random resize sizes: {}'.format(self.using_random_resize_sizes))

        # Ensure the crop shapes satisfy the div2 constraints - this is a strict requirement
        if self.using_random_crop_sizes:
            for crop_shape in self._crop_shapes:
                if crop_shape is not None:
                    if dataset_utils.count_trailing_zeroes(crop_shape[0]) < self.div2_constraint or \
                       dataset_utils.count_trailing_zeroes(crop_shape[1]) < self.div2_constraint:
                        raise ValueError('A crop shape {} does not satisfy the div2 constraint of {}'.format(crop_shape, self.div2_constraint))
        elif self._crop_shapes is not None and not self.using_random_crop_sizes:
            if dataset_utils.count_trailing_zeroes(self._crop_shapes[0]) < self.div2_constraint or \
               dataset_utils.count_trailing_zeroes(self._crop_shapes[1]) < self.div2_constraint:
                raise ValueError('A crop shape {} does not satisfy the div2 constraint of {}'.format(self._crop_shapes, self.div2_constraint))

        # Ensure the resize shapes satisfy the div2 constraints - this is not a strict requirement because crops might still fix the issue
        if self.using_random_resize_sizes:
            for resize_shape in self._resize_shapes:
                if resize_shape is not None:
                    if (resize_shape[0] is not None and dataset_utils.count_trailing_zeroes(resize_shape[0]) < self.div2_constraint) or \
                       (resize_shape[1] is not None and dataset_utils.count_trailing_zeroes(resize_shape[1]) < self.div2_constraint):
                        raise ValueError('A resize shape {} does not satisfy the div2 constraint of {}'.format(resize_shape, self.div2_constraint))
        elif self._resize_shapes is not None and not self.using_random_crop_sizes:
            if (self._resize_shapes[0] is not None and dataset_utils.count_trailing_zeroes(self._resize_shapes[0]) < self.div2_constraint) or \
               (self._resize_shapes[1] is not None and dataset_utils.count_trailing_zeroes(self._resize_shapes[1]) < self.div2_constraint):
                raise ValueError('A resize shape {} does not satisfy the div2 constraint of {}'.format(self._resize_shapes, self.div2_constraint))

    @property
    def uuid(self):
        # type: () -> int
        if not hasattr(self, '_uuid') or self._uuid is None:
            self._uuid = _get_next_uuid()
        return self._uuid

    @property
    def logger(self):
        # type: () -> Logger
        if not hasattr(self, '_logger') or self._logger is None:
            self._logger = Logger.instance()
        return self._logger

    def get_batch_crop_shape(self):
        if self.using_random_crop_sizes:
            shape = self._crop_shapes[np.random.randint(0, len(self._crop_shapes))]
            assert isinstance(shape, list) or isinstance(shape, tuple)
            return shape

        return self._crop_shapes

    def get_batch_resize_shape(self):
        if self.using_random_resize_sizes:
            shape = self._resize_shapes[np.random.randint(0, len(self._resize_shapes))]
            assert shape is None or isinstance(shape, list) or isinstance(shape, tuple)
            return shape

        return self._resize_shapes

    @abstractproperty
    def using_unlabeled_data(self):
        # type: () -> bool

        """
        Returns whether the DataGenerator is using unlabeled data.

        # Arguments
            None
        # Returns
            :return: true if using unlabeled data false otherwise
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

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
    def get_data_batch(self, step_idx, labeled_batch, unlabeled_batch):
        # type: (int, list, list) -> (np.ndarray, np.ndarray)

        """
        Returns a batch of data as numpy arrays. Either a tuple of (X, Y) or (X, Y, SW).

        # Arguments
            :param step_idx: global step index
            :param labeled_batch: index array describing the labeled data in the batch
            :param unlabeled_batch: index array describing the unlabeled data in the batch
        # Returns
            :return: A batch of data
        """

        # Sanity check for unlabeled data usage
        if not self.using_unlabeled_data and unlabeled_batch is not None:
            if len(unlabeled_batch) != 0:
                self.logger.warn('Not using unlabeled data but unlabeled batch indices were provided')

        if self.using_unlabeled_data:
            if unlabeled_batch is None or len(unlabeled_batch) == 0:
                self.logger.warn('Using unlabeled data but no unlabeled data indices were provided')

        # Random seed - use the initial random seed and the global step
        np.random.seed(self.random_seed + step_idx)
        random.seed(self.random_seed + step_idx)

    @abstractmethod
    def get_data_set_iterator(self):
        # type: () -> DataSetIterator

        """
        Returns a new data set iterator which can be passed to a Keras generator function.
        The iterator is always a new iterator and the reference is not stored within the
        DataGenerator.

        # Arguments
            None
        # Returns
            :return: a new iterator to the data set
        """

        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    def _resize_image(self, np_image, resize_shape, cval, interp):
        # type: (np.ndarray, tuple, np.ndarray, str) -> np.ndarray

        # If the resize shape is None just return the original image
        if resize_shape is None:
            return np_image

        assert(isinstance(resize_shape, list) or isinstance(resize_shape, tuple) or isinstance(resize_shape, np.ndarray))
        assert(len(resize_shape) == 2)
        assert(not (resize_shape[0] is None and resize_shape[1] is None))

        # Scale to match the sdim found in index 0
        if resize_shape[0] is not None and resize_shape[1] is None:
            target_sdim = resize_shape[0]
            img_sdim = min(np_image.shape[:2])

            if target_sdim == img_sdim:
                return np_image

            scale_factor = float(target_sdim)/float(img_sdim)
            target_shape = tuple(np.round(np.array(np_image.shape[:2], dtype=np.float32) * scale_factor).astype(dtype=np.int32))
        # Scale the match the bdim found in index 1
        elif resize_shape[0] is None and resize_shape[1] is not None:
            target_bdim = resize_shape[1]
            img_bdim = max(np_image.shape[:2])

            if target_bdim == img_bdim:
                return np_image

            scale_factor = float(target_bdim)/float(img_bdim)
            target_shape = tuple(np.round(np.array(np_image.shape[:2], dtype=np.float32) * scale_factor).astype(dtype=np.int32))
        # Scale to the exact shape
        else:
            target_shape = tuple(resize_shape)

            if target_shape[0] == np_image.shape[0] and target_shape[1] == np_image.shape[1]:
                return np_image

        return image_utils.np_resize_image_with_padding(np_image, shape=target_shape, cval=cval, interp=interp)

    def _fit_image_to_div2_constraint(self, np_image, cval, interp):
        # type: (np.ndarray, np.ndarray, str) -> np.ndarray

        # Make sure the image dimensions satisfy the div2_constraint i.e. are n times divisible
        # by 2 to work with the network. If the dimensions are not ok pad the images.
        img_height_div2 = dataset_utils.count_trailing_zeroes(np_image.shape[0])
        img_width_div2 = dataset_utils.count_trailing_zeroes(np_image.shape[1])

        if img_height_div2 < self.div2_constraint or img_width_div2 < self.div2_constraint:
            target_shape = dataset_utils.get_required_image_dimensions(np_image.shape, self.div2_constraint)
            np_image = image_utils.np_resize_image_with_padding(np_image, shape=target_shape, cval=cval, interp=interp)

        return np_image

    def _normalize_image_batch(self, batch):
        # type: (np.ndarray) -> np.ndarray

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

    def _should_apply_augmentation(self, step_idx):
        # type: () -> bool

        """
        Returns a boolean describing whether to apply augmentation.

        # Arguments
            :param step_idx: the global step index
        # Returns
            :return: true if augmentation should be applied, false otherwise
        """

        return self.use_data_augmentation and np.random.random() <= self.data_augmentation_params.augmentation_probability_function(step_idx)

    def _apply_data_augmentation_to_images(self, images, cvals, interpolations, transform_origin=None, override_channel_shift_ranges=None, override_gamma_adjust_ranges=None):
        # type: (list, list, list, np.ndarray, list, list) -> (list, ImageTransform)

        """
        Applies (same) data augmentation as per stored DataAugmentationParameters to a list of images.

        # Arguments
            :param images: images to augment (same augmentation on all)
            :param cvals: constant fill values for the images
            :param interpolations: interpolations for the images
            :param transform_origin: the origin for the transformations, if None the center of the images is used
            :param override_channel_shift_ranges: optional channel shift range override (e.g. for mask images), otherwise DataAugmentationParam used for all
            :param override_gamma_adjust_ranges: optional gamma adjust range override (e.g. for mask images), otherwise DataAugmentationParam used for all
        # Returns
            :return: the list of augmented images and ImageTransform as a tuple (images, transform)
        """

        if not self.use_data_augmentation:
            self.logger.warn('Apply data augmentation called but use_data_augmentation is false')

        num_images = len(images)
        channel_shift_ranges = [self.data_augmentation_params.channel_shift_range] * num_images if override_channel_shift_ranges is None else override_channel_shift_ranges
        gamma_adjust_ranges = [self.data_augmentation_params.gamma_adjust_range] * num_images if override_gamma_adjust_ranges is None else override_gamma_adjust_ranges

        stime = time.time()
        images, transform = image_utils.np_apply_random_transform(images=images,
                                                                  cvals=cvals,
                                                                  fill_mode=self.data_augmentation_params.fill_mode,
                                                                  interpolations=interpolations,
                                                                  transform_origin=transform_origin,
                                                                  img_data_format=self.img_data_format,
                                                                  rotation_range=self.data_augmentation_params.rotation_range,
                                                                  zoom_range=self.data_augmentation_params.zoom_range,
                                                                  width_shift_range=self.data_augmentation_params.width_shift_range,
                                                                  height_shift_range=self.data_augmentation_params.height_shift_range,
                                                                  channel_shift_ranges=channel_shift_ranges,
                                                                  horizontal_flip=self.data_augmentation_params.horizontal_flip,
                                                                  vertical_flip=self.data_augmentation_params.vertical_flip,
                                                                  gamma_adjust_ranges=gamma_adjust_ranges)

        self.logger.debug_log('Data augmentation took: {} sec'.format(time.time() - stime))

        return images, transform

    def _get_mean_teacher_data_from_image_batch(self, X):
        # type: (np.ndarray) -> np.ndarray

        if not self.generate_mean_teacher_data:
            raise ValueError('Request to get mean teacher data when generate_mean_teacher_data is False')

        # If not applying any noise - return the same images do not copy
        if not self.data_augmentation_params.using_mean_teacher_noise:
            return X

        # Copy the image batch
        teacher_img_batch = np.array(X, copy=True)

        # Apply noise transformations individually to each image
        # * Horizontal flips
        # * Vertical flips
        # * Translations
        # * Intensity shifts
        # * Gaussian noise
        noise_params = self.data_augmentation_params.mean_teacher_noise_params
        batch_size = teacher_img_batch.shape[0]

        Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')(
            delayed(pickle_method)(self, '_apply_mean_teacher_noise_to_image', teacher_img_batch=teacher_img_batch, i=i) for i in range(batch_size))

        # Apply gaussian noise to the whole batch at once
        gaussian_noise_stddev = noise_params.get('gaussian_noise_stddev')

        if gaussian_noise_stddev is not None:
            teacher_img_batch = teacher_img_batch + np.random.normal(loc=0.0, scale=gaussian_noise_stddev, size=teacher_img_batch.shape)

        return teacher_img_batch

    def _apply_mean_teacher_noise_to_image(self, teacher_img_batch, i):
        noise_params = self.data_augmentation_params.mean_teacher_noise_params

        # Figure out the correct axes according to image data format
        if self.img_data_format == 'channels_first':
            img_channel_axis = 0
            img_row_axis = 1
            img_col_axis = 2
        elif self.img_data_format == 'channels_last':
            img_row_axis = 0
            img_col_axis = 1
            img_channel_axis = 2
        else:
            raise ValueError('Unknown image data format: {}'.format(self.img_data_format))

        # Apply brightness shift
        brightness_shift_range = noise_params.get('brightness_shift_range')

        if brightness_shift_range is not None:
            brightness_shift = np.random.uniform(-brightness_shift_range, brightness_shift_range)
            teacher_img_batch[i] = teacher_img_batch[i] + brightness_shift

        # Apply horizontal flips
        horizontal_flip_probability = noise_params.get('horizontal_flip_probability')

        if horizontal_flip_probability is not None:
            if np.random.random() < horizontal_flip_probability:
                teacher_img_batch[i] = teacher_img_batch[i].swapaxes(img_col_axis, 0)
                teacher_img_batch[i] = teacher_img_batch[i][::-1, ...]
                teacher_img_batch[i] = teacher_img_batch[i].swapaxes(0, img_col_axis)

        # Apply vertical flips
        vertical_flip_probability = noise_params.get('vertical_flip_probability')

        if vertical_flip_probability is not None:
            if np.random.random() < vertical_flip_probability:
                teacher_img_batch[i] = teacher_img_batch[i].swapaxes(img_row_axis, 0)
                teacher_img_batch[i] = teacher_img_batch[i][::-1, ...]
                teacher_img_batch[i] = teacher_img_batch[i].swapaxes(0, img_row_axis)

        # Apply translations
        shift_range = noise_params.get('shift_range')

        if shift_range is not None:
            if len(shift_range) != 2:
                raise ValueError('Shift range should be a list of two values (x, y), got: {}'.format(shift_range))

            x_shift = int(float(shift_range[0] * teacher_img_batch[i].shape[img_col_axis]) * np.random.random())
            y_shift = int(float(shift_range[1] * teacher_img_batch[i].shape[img_row_axis]) * np.random.random())
            shift_val = [y_shift, x_shift, 0]
            temp_cval = -900.0
            shift(input=teacher_img_batch[i], shift=shift_val, output=teacher_img_batch[i], order=3, mode='constant', cval=temp_cval)

            # Replace the invalid constant values from the output
            if img_channel_axis == 2:
                cval_mask = teacher_img_batch[i][:, :, 0] == temp_cval
                teacher_img_batch[i][cval_mask] = self.per_channel_mean
            else:
                cval_mask = teacher_img_batch[i][0, :, :] == temp_cval
                teacher_img_batch[i][cval_mask] = self.per_channel_mean

#######################################
# SEGMENTATION DATA GENERATOR
#######################################


class SegmentationDataGenerator(DataGenerator):

    UUID_TO_LABELED_DATA_SET = {}
    UUID_TO_UNLABELED_DATA_SET = {}

    def __init__(self,
                 labeled_data_set,
                 unlabeled_data_set,
                 num_labeled_per_batch,
                 num_unlabeled_per_batch,
                 batch_data_format,
                 params,
                 class_weights=None,
                 label_generation_function_type=SuperpixelSegmentationFunctionType.NONE):
        # type: (LabeledImageDataSet, UnlabeledImageDataSet, int, int, BatchDataFormat, SegmentationDataGeneratorParameters, np.array[np.float32], SuperpixelSegmentationFunctionType) -> None

        """
        # Arguments
            :param labeled_data_set: LabeledImageSet instance of the labeled data set
            :param unlabeled_data_set: UnlabeledImageSet instance of the unlabeled data set
            :param num_labeled_per_batch: number of labeled images per batch
            :param num_unlabeled_per_batch: number of unlabeled images per batch
            :param batch_data_format: format of the data batches
            :param params: SegmentationDataGeneratorParameters object
            :param class_weights: class weights
            :param label_generation_function_type: function type for the superpixel label generation
        """

        self.labeled_data_set = labeled_data_set
        self.unlabeled_data_set = unlabeled_data_set

        self.num_labeled_per_batch = num_labeled_per_batch
        self.num_unlabeled_per_batch = num_unlabeled_per_batch

        # Unwrap segmentation data generator specific parameters
        self.material_class_information = params.material_class_information
        self.use_material_samples = params.use_material_samples
        self.use_selective_attention = params.use_selective_attention
        self.use_adaptive_sampling = params.use_adaptive_sampling
        self.mask_cval = params.mask_cval
        self.num_crop_reattempts = params.num_crop_reattempts
        self.num_classes = len(self.material_class_information)

        super(SegmentationDataGenerator, self).__init__(batch_data_format, params)

        # Use black (background)
        if self.mask_cval is None:
            self.mask_cval = np.array([0.0] * 3, dtype=np.float32)
            self.logger.log('SegmentationDataGenerator: Using mask cval: {}'.format(list(self.mask_cval)))

        # Sanity check to ensure that MaterialSamples are part of the data set if used
        if self.use_material_samples:
            if self.labeled_data_set.material_samples is None or len(self.labeled_data_set.material_samples) == 0:
                raise ValueError('Use material samples is true, but labeled data set does not contain material samples')

        if labeled_data_set is None:
            raise ValueError('SegmentationDataGenerator does not support empty labeled data set')

        if class_weights is None:
            raise ValueError('Class weights is None. Use a numpy array of ones instead of None')

        self.class_weights = class_weights
        self.label_generation_function_type = label_generation_function_type

        self.logger.log('Use material samples: {}'.format(self.use_material_samples))
        self.logger.log('Use selective attention: {}'.format(self.use_selective_attention))
        self.logger.log('Use adaptive sampling: {}'.format(self.use_adaptive_sampling))

        if self.use_selective_attention and not self.use_material_samples:
            raise ValueError('Selective attention can only be used with material samples - enable material samples')

        if self.use_adaptive_sampling and not self.use_material_samples:
            raise ValueError('Adaptive sampling can only be used with material samples - enable material samples')

    @property
    def labeled_data_set(self):
        # type: () -> LabeledImageDataSet
        return SegmentationDataGenerator.UUID_TO_LABELED_DATA_SET.get(self.uuid)

    @labeled_data_set.setter
    def labeled_data_set(self, data_set):
        # type: (LabeledImageDataSet) -> None
        SegmentationDataGenerator.UUID_TO_LABELED_DATA_SET[self.uuid] = data_set

    @property
    def unlabeled_data_set(self):
        # type: () -> UnlabeledImageDataSet
        return SegmentationDataGenerator.UUID_TO_UNLABELED_DATA_SET.get(self.uuid)

    @unlabeled_data_set.setter
    def unlabeled_data_set(self, data_set):
        # type: (UnlabeledImageDataSet) -> None
        SegmentationDataGenerator.UUID_TO_UNLABELED_DATA_SET[self.uuid] = data_set

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
               self.num_unlabeled_per_batch > 0 and \
               self.batch_data_format != BatchDataFormat.SUPERVISED

    def get_data_set_iterator(self):
        # type: () -> DataSetIterator

        """
        Returns a new data set iterator which can be passed to a Keras generator function.
        The iterator is always a new iterator and the reference is not stored within the
        DataGenerator.

        # Arguments
            None
        # Returns
            :return: a new iterator to the data set
        """

        if self.use_material_samples:
            return MaterialSampleDataSetIterator(
                data_generator=self,
                material_samples=self.labeled_data_set.material_samples,
                n_unlabeled=self.unlabeled_data_set.size if self.using_unlabeled_data else 0,
                labeled_batch_size=self.num_labeled_per_batch,
                unlabeled_batch_size=self.num_unlabeled_per_batch if self.using_unlabeled_data else 0,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed,
                initial_epoch=self.initial_epoch,
                iteration_mode=MaterialSampleIterationMode.UNIFORM_MEAN,
                balance_pixel_samples=self.use_adaptive_sampling)
        else:
            return BasicDataSetIterator(
                data_generator=self,
                n_labeled=self.labeled_data_set.size,
                labeled_batch_size=self.num_labeled_per_batch,
                n_unlabeled=self.unlabeled_data_set.size if self.using_unlabeled_data else 0,
                unlabeled_batch_size=self.num_unlabeled_per_batch if self.using_unlabeled_data else 0,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed,
                initial_epoch=self.initial_epoch)

    def get_labeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        # type: (int, np.array[int], list, list) -> (list[np.array], list[np.array])

        """
        # Arguments
            :param step_index: current step index
            :param index_array: indices of the labeled data
            :param crop_shape: crop shape or None
            :param resize_shape: resize shape or None
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
                                                                mask_type=SegmentationMaskEncodingType.INDEX,
                                                                crop_shape=crop_shape,
                                                                resize_shape=resize_shape) for i in range(len(labeled_batch_files))]

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

    def get_unlabeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        # type: (int, np.array[int], list, list) -> (list[np.array], list[np.array], list[np.array])

        """
        # Arguments
            :param step_index: index of the current step
            :param index_array: indices of the unlabeled data
            :param crop_shape: crop shape or None
            :param resize_shape: resize shape or None
        # Returns
            :return: unlabeled data as three lists: (X, Y, WEIGHTS)
        """

        # If we don't have unlabeled data return two empty lists
        if not self.using_unlabeled_data:
            return [], [], []

        unlabeled_batch_files = self.unlabeled_data_set.get_indices(index_array)

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        unlabeled_data = [self.get_unlabeled_segmentation_data_pair(step_idx=step_index, photo_file=photo_file, crop_shape=crop_shape, resize_shape=resize_shape) for photo_file in unlabeled_batch_files]
        X_unlabeled, Y_unlabeled = zip(*unlabeled_data)
        W_unlabeled = []

        for y in Y_unlabeled:
            W_unlabeled.append(np.ones_like(y, dtype=np.float32))

        return list(X_unlabeled), list(Y_unlabeled), W_unlabeled

    def get_labeled_segmentation_data_pair(self, step_idx, photo_file, mask_file, material_sample, crop_shape, resize_shape, mask_type=SegmentationMaskEncodingType.INDEX):
        # type: (int, ImageFile, ImageFile, MaterialSample, list, list, SegmentationMaskEncodingType) -> (np.array, np.array)

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
        np_photo, np_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx,
                                                                      np_photo=np_photo,
                                                                      np_mask=np_mask,
                                                                      photo_cval=self.photo_cval,
                                                                      mask_cval=self.mask_cval,
                                                                      material_sample=material_sample,
                                                                      crop_shape=crop_shape,
                                                                      resize_shape=resize_shape,
                                                                      retry_crops=True)

        # Only do it if the material is actually found - otherwise print a warning
        if material_sample is not None:
            material_pixels_mask = np.equal(np_mask[:, :, 0], material_sample.material_r_color)

            # Sanity check: material samples are supposed to guarantee material instances
            if not np.any(material_pixels_mask):
                self.logger.log_image(np_photo, file_name='{}_crop_missing_{}.jpg'.format(photo_file.file_name, material_sample.material_id))
                self.logger.warn('Material sample for material id {} was given but no corresponding entries were found in the cropped mask. Found r colors: {}'
                                 .format(material_sample.material_id, list(np.unique(np_mask[:, :, 0]))))
            else:
                # If we are using selective attention mark everything else as zero (background) besides
                # the red channel representing the current material sample
                if self.use_selective_attention:
                    np_mask[np.logical_not(material_pixels_mask)] = 0

        # Expand the mask image to the one-hot encoded shape: H x W x NUM_CLASSES
        if mask_type == SegmentationMaskEncodingType.ONE_HOT:
            np_mask = dataset_utils.one_hot_encode_mask(np_mask, self.material_class_information)
        elif mask_type == SegmentationMaskEncodingType.INDEX:
            np_mask = dataset_utils.index_encode_mask(np_mask, self.material_class_information)
        else:
            raise ValueError('Unknown mask_type: {}'.format(mask_type))

        return np_photo, np_mask

    def get_unlabeled_segmentation_data_pair(self, step_idx, photo_file, crop_shape, resize_shape):
        # type: (int, ImageFile, list, list) -> (np.array, np.array)

        """
        Returns a photo mask pair for semi-supervised/unsupervised segmentation training.
        Will apply data augmentation and cropping as instructed in the parameters.

        The photos are not normalized to range [-1,1] within the function.

        # Arguments
            :param step_idx: index of the current training step
            :param photo_file: an ImageFile of the photo
            :param crop_shape: crop shape or None
            :param resize_shape: resize shape or None
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        # Load the photo as PIL image
        photo_file = photo_file.get_image(color_channels=self.num_color_channels)
        np_photo = img_to_array(photo_file)

        # Generate mask for the photo - note: the labels are generated before cropping
        # and augmentation to capture global structure within the image
        np_mask = self._generate_mask_for_unlabeled_image(np_photo)

        # Expand the last dimension of the mask to make it compatible with augmentation functions
        np_mask = np_mask[:, :, np.newaxis]

        # Apply crops and augmentation
        np_photo, np_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx,
                                                                      np_photo=np_photo,
                                                                      np_mask=np_mask,
                                                                      photo_cval=self.photo_cval,
                                                                      mask_cval=[0],
                                                                      crop_shape=crop_shape,
                                                                      resize_shape=resize_shape,
                                                                      retry_crops=False)

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

    def process_segmentation_photo_mask_pair(self, step_idx, np_photo, np_mask, photo_cval, mask_cval, material_sample=None, crop_shape=None, resize_shape=None, retry_crops=True):
        # type: (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MaterialSample, list, list, bool) -> (np.ndarray, np.ndarray)

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
            :param crop_shape: shape of the crop or None
            :param resize_shape: shape of the target resize or None
            :param retry_crops: retries crops if the whole crop is 0 (BG)
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        if np_photo.shape[:2] != np_mask.shape[:2]:
            raise ValueError('Non-matching photo and mask shapes: {} != {}'.format(np_photo.shape, np_mask.shape))

        if material_sample is not None and crop_shape is None:
            raise ValueError('Cannot use material samples without cropping')

        # Check whether we need to resize the photo and the mask to a constant size
        if resize_shape is not None:
            np_photo = self._resize_image(np_photo, resize_shape=resize_shape, cval=photo_cval, interp='bicubic')
            np_mask = self._resize_image(np_mask, resize_shape=resize_shape, cval=mask_cval, interp='nearest')

        # Check whether any of the image dimensions is smaller than the crop,
        # if so pad with the assigned fill colors
        if crop_shape is not None and (np_photo.shape[0] < crop_shape[0] or np_photo.shape[1] < crop_shape[1]):
            # Image dimensions must be at minimum the same as the crop dimension
            # on each axis. The photo needs to be filled with the photo_cval color and masks
            # with the mask cval color
            min_img_shape = (max(crop_shape[0], np_photo.shape[0]), max(crop_shape[1], np_photo.shape[1]))
            np_photo = image_utils.np_pad_image_to_shape(np_photo, min_img_shape, photo_cval)
            np_mask = image_utils.np_pad_image_to_shape(np_mask, min_img_shape, mask_cval)

        # If we are using data augmentation apply the random transformation
        # to both the image and mask now. We apply the transformation to the
        # whole image to decrease the number of 'dead' pixels due to transformations
        # within the possible crop.
        bbox = material_sample.get_bbox_rel() if material_sample is not None else None

        if self._should_apply_augmentation(step_idx):
            np_orig_photo = None
            np_orig_mask = None

            if material_sample is not None:
                np_orig_photo = np.array(np_photo, copy=True)
                np_orig_mask = np.array(np_mask, copy=True)

            images, transform = self._apply_data_augmentation_to_images(images=[np_photo, np_mask],
                                                                        cvals=[photo_cval, mask_cval],
                                                                        interpolations=[ImageInterpolation.BICUBIC, ImageInterpolation.NEAREST],
                                                                        override_channel_shift_ranges=[self.data_augmentation_params.channel_shift_range, None],
                                                                        override_gamma_adjust_ranges=[self.data_augmentation_params.gamma_adjust_range, None])
            # Unpack the images
            np_photo, np_mask = images

            if material_sample is not None:
                material_lost_during_transform = not np.any(np.equal(np_mask, material_sample.material_r_color))

                if not material_lost_during_transform:
                    # Applies the same transform the bounding box coordinates
                    bbox = self.transform_bbox(bbox, transform, np_mask, material_sample)

                # If the material was lost during the random transform - revert to unaugmented values or
                # If we cannot build a new valid axis-aligned bounding box - revert to unaugmented values
                if material_lost_during_transform or bbox is None:
                    bbox = material_sample.get_bbox_rel()
                    np_photo = np_orig_photo
                    np_mask = np_orig_mask
                    self.logger.debug_log('Reverting to original bbox and image data, material lost: {}, cannot rebuild axis-aligned bbox: {}'
                                          .format(material_lost_during_transform, bbox is None))

        # If a crop size is given: take a random crop of both the image and the mask
        if crop_shape is not None:
            for attempt in xrange(max(self.num_crop_reattempts+1, 1), 0, -1):
                if material_sample is None:
                    # If we don't have a bounding box as a hint for cropping - take random crops
                    x1y1, x2y2 = image_utils.np_get_random_crop_area(np_mask, crop_shape[1], crop_shape[0])
                else:
                    # Use the bounding box information to take a targeted crop
                    x1y1, x2y2 = self.get_random_bbox_crop_area(bbox=bbox,
                                                                img_shape=np_mask.shape[:2],
                                                                crop_shape=crop_shape,
                                                                material_sample=material_sample)

                mask_crop = image_utils.np_crop_image(np_mask, x1y1[0], x1y1[1], x2y2[0], x2y2[1])

                if material_sample is None:
                    # If the mask has something else besides background
                    valid_crop_found = not np.all(np.equal(mask_crop[:, :, 0], 0))

                    # Warn if the crop contains only zero and the mask is not all zero
                    if not valid_crop_found and not np.all(np.equal(np_mask, 0)):
                        self.logger.warn('Only background found within crop area of shape: {}'.format(crop_shape))
                else:
                    # If the mask contains pixels of the desired material
                    valid_crop_found = np.any(np.equal(mask_crop[:, :, 0], material_sample.material_r_color))

                    if not valid_crop_found:
                        self.logger.warn('Material not found within crop area of shape: {} for material id: {} and material red color: {}'
                                         .format(crop_shape, material_sample.material_id, material_sample.material_r_color))

                # If a valid crop was found or this is the last attempt or we should not retry crops
                stop_iteration = valid_crop_found or attempt-1 <= 0 or not retry_crops

                if stop_iteration:
                    np_mask = mask_crop
                    np_photo = image_utils.np_crop_image(np_photo, x1=x1y1[0], y1=x1y1[1], x2=x2y2[0], y2=x2y2[1])
                    break

        # Make sure both satisfy the div2 constraint
        np_photo = self._fit_image_to_div2_constraint(np_image=np_photo, cval=photo_cval, interp='bicubic')
        np_mask = self._fit_image_to_div2_constraint(np_image=np_mask, cval=mask_cval, interp='nearest')

        return np_photo, np_mask

    def get_random_bbox_crop_area(self, bbox, img_shape, crop_shape, material_sample):
        tlc, trc, brc, blc = bbox
        crop_height, crop_width = crop_shape
        img_height, img_width = img_shape

        # Transform bbox to absolute coordinates from relative
        bbox_ymin = int(np.round(tlc[0]*img_height))
        bbox_ymax = min(int(np.round(brc[0]*img_height)) + 1, img_height)  # +1 because the bbox ymax is inclusive
        bbox_xmin = int(np.round(tlc[1]*img_width))
        bbox_xmax = min(int(np.round(brc[1]*img_width)) + 1, img_width)    # +1 because the bbox xmax is inclusive
        bbox_height = bbox_ymax - bbox_ymin
        bbox_width = bbox_xmax - bbox_xmin

        # Apparently the material sample data can have single pixel height/width bounding boxes
        # expand the bounding box a bit to avoid errors
        if bbox_height <= 0:
            self.logger.debug_log('Invalid bounding box height: mat id: {}, file name: {}, bbox: {}, original bbox: {} - inflating'
                                  .format(material_sample.material_id, material_sample.file_name, bbox, material_sample.get_bbox_rel()))
            bbox_ymin = max(bbox_ymin - 1, 0)
            bbox_ymax = min(bbox_ymax + 1, img_height)
            bbox_height = bbox_ymax - bbox_ymin

        if bbox_width <= 0:
            self.logger.debug_log('Invalid bounding box width: mat id: {}, file name: {}, bbox: {}, original bbox: {} - inflating'
                                  .format(material_sample.material_id, material_sample.file_name, bbox, material_sample.get_bbox_rel()))
            bbox_xmin = max(bbox_xmin - 1, 0)
            bbox_xmax = min(bbox_xmax + 1, img_width)
            bbox_width = bbox_xmax - bbox_xmin

        # If after the fix bounding box is still of invalid size throw an error
        if bbox_height <= 0 or bbox_width <= 0:
            raise ValueError('Invalid bounding box dimensions: mat id: {}, file name: {}, bbox: {}, original bbox: {}'
                             .format(material_sample.material_id, material_sample.file_name, bbox, material_sample.get_bbox_rel()))

        # Calculate the difference in height and width between the bbox and crop
        height_diff = abs(crop_height - bbox_height)
        width_diff = abs(crop_width - bbox_width)

        # If the crop can fit the whole material sample within it
        if bbox_height <= crop_height and bbox_width <= crop_width:
            crop_ymin = bbox_ymin - np.random.randint(0, min(height_diff + 1, bbox_ymin + 1))
            crop_xmin = bbox_xmin - np.random.randint(0, min(width_diff + 1, bbox_xmin + 1))
        # If the bounding box is bigger than the crop in both width and height
        elif bbox_height > crop_height and bbox_width > crop_width:
            crop_ymin = bbox_ymin + np.random.randint(0, height_diff + 1)
            crop_xmin = bbox_xmin + np.random.randint(0, width_diff + 1)
        # If the crop width is smaller than the bbox width
        elif bbox_width > crop_width:
            crop_ymin = bbox_ymin - np.random.randint(0, min(height_diff + 1, bbox_ymin + 1))
            crop_xmin = bbox_xmin + np.random.randint(0, width_diff + 1)
        # If the crop height is smaller than the bbox height
        elif bbox_height > crop_height:
            crop_ymin = bbox_ymin + np.random.randint(0, height_diff + 1)
            crop_xmin = bbox_xmin - np.random.randint(0, min(width_diff + 1, bbox_xmin + 1))
        else:
            raise ValueError('Unable to determine crop y_min and x_min')

        crop_ymax = crop_ymin + crop_height
        crop_xmax = crop_xmin + crop_width

        # Sanity check for y values
        if crop_ymax > img_height:
            diff = crop_ymax - img_height
            crop_ymin = crop_ymin - diff
            crop_ymax = crop_ymax - diff

        # Sanity check for x values
        if crop_xmax > img_width:
            diff = crop_xmax - img_width
            crop_xmin = crop_xmin - diff
            crop_xmax = crop_xmax - diff

        return (crop_xmin, crop_ymin), (crop_xmax, crop_ymax)

    def transform_bbox(self, bbox, transform, np_mask, material_sample):
        # type: (tuple[tuple[float, float]], ImageTransform, np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]

        # Transform the bbox coords to ndarray
        # Bbox is: tlc, trc, brc, blc
        original_bbox_coords = np.array(bbox, dtype=np.float32)

        # Sanity check the bbox dimensions
        if original_bbox_coords.shape[0] != 4 and original_bbox_coords.shape[1] != 2:
            raise ValueError('Expected bounding box with dimensions [4,2] got shape: {}'.format(original_bbox_coords.shape))

        # The transform needs to get the coordinates in (x,y) instead of (y,x), so flip the coordinates there and back
        original_bbox_coords = np.fliplr(original_bbox_coords)
        transformed_bbox_coords = np.fliplr(transform.transform_normalized_coordinates(original_bbox_coords))

        # Switch to absolute coordinates
        transformed_bbox_coords[:, 0] *= np_mask.shape[0]
        transformed_bbox_coords[:, 1] *= np_mask.shape[1]
        transformed_bbox_coords = np.round(transformed_bbox_coords).astype(dtype=np.int32)

        # Clamp the values of the corners into valid ranges [[0, img_height], [0, img_width]]
        tf_y_min = np.clip(np.min(transformed_bbox_coords[:, 0]), 0, transform.image_height)
        tf_y_max = np.clip(np.max(transformed_bbox_coords[:, 0]), 0, transform.image_height)
        tf_x_min = np.clip(np.min(transformed_bbox_coords[:, 1]), 0, transform.image_width)
        tf_x_max = np.clip(np.max(transformed_bbox_coords[:, 1]), 0, transform.image_width)

        # If the area is less than 4 pixels return None
        if tf_y_max - tf_y_min <= 2 or tf_x_max - tf_x_min <= 2:
            return None

        # Sanity check that the material is found within the bounding box
        # Note: it is possible that there are other instances of the same material present in the image unrelated to this bbox
        if not np.any(np.equal(np_mask[tf_y_min:tf_y_max, tf_x_min:tf_x_max, 0], material_sample.material_r_color)):
            material_img_instances = np.transpose(np.where(np.equal(np_mask[:, :, 0], material_sample.material_r_color)))
            self.logger.debug_log('Transformed bbox did not contain material {}. Original bbox: ({}, {}). Transformed bbox: ({}, {}). Image shape: {}. Image material instances: {}'
                                  .format(material_sample.material_id, bbox[0], bbox[2], (tf_y_min, tf_x_min), (tf_y_max, tf_x_max), np_mask.shape, material_img_instances))
            return None

        # Refine the values - find the area within the bounding box that actually contains the material
        # Note: the coordinates returned by np.where are from within the transformed bounding box so add tf_y_min and tf_x_min
        tf_material_y_coords, tf_material_x_coords = np.where(np.equal(np_mask[tf_y_min:tf_y_max, tf_x_min:tf_x_max, 0], material_sample.material_r_color))
        tf_material_y_coords += tf_y_min
        tf_material_x_coords += tf_x_min

        rf_y_min = np.min(tf_material_y_coords)
        rf_y_max = min(np.max(tf_material_y_coords)+1, transform.image_height)
        rf_x_min = np.min(tf_material_x_coords)
        rf_x_max = min(np.max(tf_material_x_coords)+1, transform.image_width)

        # If the area is less than 4 pixels return None
        if rf_y_max - rf_y_min <= 2 or rf_x_max - rf_x_min <= 2:
            return None

        # Final sanity check that the material is found within the bounding box
        # This shouldn't be possible without a bug in the refinement code because the material was already in the transformed bbox
        if not np.any(np.equal(np_mask[rf_y_min:rf_y_max, rf_x_min:rf_x_max, 0], material_sample.material_r_color)):
            material_img_instances = np.transpose(np.where(np.equal(np_mask[:, :, 0], material_sample.material_r_color)))
            self.logger.warn('Refined bbox did not contain material {}. Original bbox: ({}, {}). Transformed bbox: ({}, {}). Refined bbox: ({}, {}). Image shape: {}. Image material instances: {}'
                             .format(material_sample.material_id, bbox[0], bbox[2], (tf_y_min, tf_x_min), (tf_y_max, tf_x_max), (rf_y_min, rf_x_min), (rf_y_max, rf_x_max), np_mask.shape, material_img_instances))
            return None

        # Rebuild the bounding box and represent in (y, x) - use relative coordinates [0.0, 1.0]
        tlc = (float(rf_y_min)/np_mask.shape[0], float(rf_x_min)/np_mask.shape[1])
        trc = (float(rf_y_min)/np_mask.shape[0], float(rf_x_max)/np_mask.shape[1])
        brc = (float(rf_y_max)/np_mask.shape[0], float(rf_x_max)/np_mask.shape[1])
        blc = (float(rf_y_max)/np_mask.shape[0], float(rf_x_min)/np_mask.shape[1])

        return tlc, trc, brc, blc

    def get_data_batch(self, step_idx, labeled_batch, unlabeled_batch):
        # type: (int, list[int], list[int]) -> (list[np.ndarray], list[np.ndarray])

        """
        Returns a batch of data as numpy arrays. Either a tuple of (X, Y) or (X, Y, SW).

        # Arguments
            :param step_idx: global step index
            :param labeled_batch: index array describing the labeled data in the batch
            :param unlabeled_batch: index array describing the unlabeled data in the batch
        # Returns
            :return: A batch of data
        """

        super(SegmentationDataGenerator, self).get_data_batch(step_idx, labeled_batch, unlabeled_batch)

        crop_shape = self.get_batch_crop_shape()
        resize_shape = self.get_batch_resize_shape()

        #self.logger.debug_log('Batch crop shape: {}, resize shape: {}'.format(crop_shape, resize_shape))
        #self.logger.debug_log('Generating batch data for step {}: labeled: {}, ul: {}'.format(step_idx, labeled_batch, unlabeled_batch))

        stime = time.time()
        X, Y, W = self.get_labeled_batch_data(step_idx, labeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        X_unlabeled, Y_unlabeled, W_unlabeled = self.get_unlabeled_batch_data(step_idx, unlabeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        self.logger.debug_log('Raw data generation took: {}s'.format(time.time() - stime))

        X = X + X_unlabeled
        Y = Y + Y_unlabeled
        W = W + W_unlabeled

        num_unlabeled_samples_in_batch = len(X_unlabeled)
        num_samples_in_batch = len(X)

        # Concatenate the lists to numpy arrays and ensure that correct types are used
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        W = np.asarray(W, dtype=np.float32)

        # Normalize the photo batch data
        X = self._normalize_image_batch(X)

        if self.batch_data_format == BatchDataFormat.SUPERVISED:
            batch_input_data = [X]
            batch_output_data = [np.expand_dims(Y, -1)]
        elif self.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # The dimensions of the number of unlabeled in the batch must match with batch dimension
            num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

            # Generate a dummy output for the dummy loss function and yield a batch of data
            dummy_output = np.zeros(shape=[num_samples_in_batch], dtype=np.int32)

            batch_input_data = [X, Y, W, num_unlabeled]

            # Provide the true classification masks for labeled samples only - these go to the second loss function
            # in the semi-supervised model that is only used to calculate metrics. The output has to have the same
            # rank as the output from the network
            logits_output = np.expand_dims(np.copy(Y), -1)
            logits_output[num_samples_in_batch - num_unlabeled_samples_in_batch:] = 0
            batch_output_data = [dummy_output, logits_output]
        else:
            raise ValueError('Unknown batch data format: {}'.format(self.batch_data_format))

        # Sanity check for batch dimensions
        for element in batch_input_data:
            if element.shape[0] != num_samples_in_batch:
                self.logger.warn('Invalid input data first (batch) dimension: {} should be {}'.format(element.shape[0], num_samples_in_batch))

        for element in batch_output_data:
            if element.shape[0] != num_samples_in_batch:
                self.logger.warn('Invalid output data first (batch) dimension: {} should be {}'.format(element.shape[0], num_samples_in_batch))

        self.logger.debug_log('Data generation took: {}s'.format(time.time() - stime))

        # TODO: Debug and teacher data

        # If we are in debug mode, save the batch images - this is right before the images enter
        # into the neural network
        #if settings.DEBUG:
        #    b_min = np.min(img_batch)
        #    b_max = np.max(img_batch)

        #    for i in range(0, len(img_batch)):
        #        img = ((img_batch[i] - b_min) / (b_max - b_min)) * 255.0
        #        mask = mask_batch[i][:, :, np.newaxis]*255.0
        #        self.logger.debug_log_image(img, '{}_{}_{}_photo.jpg'.format("val" if validation else "tr", step_index, i), scale=False)
        #        self.logger.debug_log_image(mask, file_name='{}_{}_{}_mask.png'.format("val" if validation else "tr", step_index, i), format='PNG')

        return batch_input_data, batch_output_data

    def _generate_mask_for_unlabeled_image(self, np_img):
        # type: (np.ndarray) -> np.ndarray

        """
        Generates labels (mask) for unlabeled images. Either uses the default generator which is
        numpy array of same shape as np_image with every labeled as 0 or the one specified during
        initialization. The label generator should encode the information in the form HxW.

        # Arguments
            :param np_img: the image as a numpy array
        # Returns
            :return:
        """

        if self.label_generation_function_type == SuperpixelSegmentationFunctionType.NONE:
            return np.zeros(shape=(np_img.shape[0], np_img.shape[1]), dtype=np.int32)
        elif self.label_generation_function_type == SuperpixelSegmentationFunctionType.FELZENSWALB:
            return image_utils.np_get_felzenswalb_segmentation(np_img, scale=700, sigma=0.6, min_size=250, normalize_img=True, borders_only=True)
        elif self.label_generation_function_type == SuperpixelSegmentationFunctionType.SLIC:
            return image_utils.np_get_slic_segmentation(np_img, n_segments=300, sigma=1, compactness=10.0, max_iter=20, normalize_img=True, borders_only=True)
        elif self.label_generation_function_type == SuperpixelSegmentationFunctionType.QUICKSHIFT:
            return image_utils.np_get_quickshift_segmentation(np_img, kernel_size=20, max_dist=15, ratio=0.5, normalize_img=True, borders_only=True)
        elif self.label_generation_function_type == SuperpixelSegmentationFunctionType.WATERSHED:
            return image_utils.np_get_watershed_segmentation(np_img, markers=250, compactness=0.001, normalize_img=True, borders_only=True)
        else:
            raise ValueError('Unknown label generation function type: {}'.format(self.label_generation_function_type))


################################################
# CLASSIFICATION DATA GENERATOR
################################################

class MINCDataSetType(Enum):
    MINC = 0
    MINC_2500 = 1


class MINCDataSet(object):

    def __init__(self, name, path_to_photo_archive, label_mappings_file_path, data_set_file_path):
        self.label_mappings_file_path = label_mappings_file_path
        self.data_set_file_path = data_set_file_path

        # Read label mappings
        self.minc_class_to_minc_label, self.minc_label_to_custom_label = self._load_label_mappings(label_mappings_file_path)

        # Read data set
        self.samples = self._load_samples(data_set_file_path, self.minc_class_to_minc_label)

        # Create the ImageSet
        file_list = [s.file_name for s in self.samples]

        # One image may contain multiple MINC samples - remove duplicate file names from the list
        file_list = list(set(file_list))

        self._image_set = ImageSet(name=name, path_to_archive=path_to_photo_archive, file_list=file_list)

    @property
    def num_classes(self):
        return len(self.minc_class_to_minc_label)

    @property
    def size(self):
        # type: () -> int
        return len(self.samples)

    @property
    def photo_image_set(self):
        return self._image_set

    def _load_samples(self, file_path, minc_class_to_minc_label):
        samples = list()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Attempt to figure out whether this is a MINC-2500 file or MINC file
        # MINC-2500: images/<class name>/<class name>_<photo id>.jpg
        # MINC: label,photo_id,x,y
        is_minc = len(lines[0].split(',')) == 4
        is_minc_2500 = not is_minc

        if is_minc:
            self.data_set_type = MINCDataSetType.MINC
            for line in lines:
                line = line.strip()
                # Each line of the file is: 4-tuple list of (label,photo_id,x,y)
                minc_label, photo_id, x, y = line.split(',')
                samples.append(MINCSample(minc_label=int(minc_label), photo_id=photo_id.strip(), x=float(x), y=float(y)))
        elif is_minc_2500:
            self.data_set_type = MINCDataSetType.MINC_2500

            for line in lines:
                line = line.strip()
                file_name = line.split('/')[-1]
                class_name = file_name.split('_')[0]
                minc_label = minc_class_to_minc_label[class_name]
                samples.append(MINCSample(minc_label=int(minc_label), photo_id=file_name, x=-1.0, y=-1.0))

        return samples

    def _load_label_mappings(self, file_path):
        minc_label_to_custom_label = dict()
        minc_class_to_minc_label = dict()

        with open(file_path, 'r') as f:
            # The first line should be skipped because it describes the data, which is
            # in the format of substance_name,minc_class_idx,custom_class_idx
            lines = f.readlines()

            for idx, line in enumerate(lines):
                # Skip the first line
                if idx == 0:
                    continue

                line = line.strip()
                substance_name, minc_class_idx, custom_class_idx = line.split(',')

                # Check that there are no duplicate entries for MINC class ids
                if minc_label_to_custom_label.has_key(int(minc_class_idx)):
                    raise ValueError('Label mapping already contains entry for MINC class id: {}'.format(int(minc_class_idx)))

                # Check that there are no duplicate entries for custom class ids
                if int(custom_class_idx) in minc_label_to_custom_label.values():
                    raise ValueError('Label mapping already contains entry for custom class id: {}'.format(int(custom_class_idx)))

                minc_label_to_custom_label[int(minc_class_idx)] = int(custom_class_idx)

                if minc_class_to_minc_label.has_key(substance_name):
                    raise ValueError('Label mapping already contains entry for MINC class name: {}'.format(substance_name))

                if int(minc_class_idx) in minc_class_to_minc_label.values():
                    raise ValueError('Label mapping already contains entry for MINC class id: {}'.format(minc_class_idx))

                minc_class_to_minc_label[substance_name] = int(minc_class_idx)

        return minc_class_to_minc_label, minc_label_to_custom_label


class ClassificationDataGenerator(DataGenerator):

    # Static values are shared across processes
    UUID_TO_LABELED_DATA_SET = {}
    UUID_TO_UNLABELED_DATA_SET = {}

    def __init__(self,
                 labeled_data_set,
                 unlabeled_data_set,
                 num_labeled_per_batch,
                 num_unlabeled_per_batch,
                 class_weights,
                 batch_data_format,
                 params):
        # type: (MINCDataSet, UnlabeledImageDataSet, int, int, np.ndarray, BatchDataFormat, DataGeneratorParameters) -> None

        self.labeled_data_set = labeled_data_set
        self.unlabeled_data_set = unlabeled_data_set
        self.num_labeled_per_batch = num_labeled_per_batch
        self.num_unlabeled_per_batch = num_unlabeled_per_batch
        self.class_weights = np.array(class_weights, dtype=np.float32)

        super(ClassificationDataGenerator, self).__init__(batch_data_format, params)

        # Create a dummy label vector for unlabeled data (one-hot) all zeros
        self.dummy_label_vector = np.zeros(self.labeled_data_set.num_classes, dtype=np.float32)

        # Sanity checks
        if self.labeled_data_set.data_set_type == MINCDataSetType.MINC_2500 and self._crop_shapes is not None:
            self.logger.warn('Using MINC-2500 data set with cropping - cropping is not applied to MINC-2500 data'.format(self._crop_shapes))

        if self.labeled_data_set.data_set_type == MINCDataSetType.MINC and self._crop_shapes is None:
            self.logger.warn('Using MINC data set without cropping or fully specified resize - is this intended?')

    @property
    def labeled_data_set(self):
        # type: () -> MINCDataSet
        return ClassificationDataGenerator.UUID_TO_LABELED_DATA_SET.get(self.uuid)

    @labeled_data_set.setter
    def labeled_data_set(self, data_set):
        # type: (MINCDataSet) -> None
        ClassificationDataGenerator.UUID_TO_LABELED_DATA_SET[self.uuid] = data_set

    @property
    def unlabeled_data_set(self):
        # type: () -> UnlabeledImageDataSet
        return ClassificationDataGenerator.UUID_TO_UNLABELED_DATA_SET.get(self.uuid)

    @unlabeled_data_set.setter
    def unlabeled_data_set(self, data_set):
        # type: (UnlabeledImageDataSet) -> None
        ClassificationDataGenerator.UUID_TO_UNLABELED_DATA_SET[self.uuid] = data_set

    def get_all_photos(self):
        photos = []
        photos += self.labeled_data_set.photo_image_set.image_files

        if self.using_unlabeled_data:
            photos += self.unlabeled_data_set.photo_image_set.image_files

        return photos

    def get_data_set_iterator(self):
        # type: () -> DataSetIterator

        """
        Returns a new data set iterator which can be passed to a Keras generator function.
        The iterator is always a new iterator and the reference is not stored within the
        DataGenerator.

        # Arguments
            None
        # Returns
            :return: a new iterator to the data set
        """

        return BasicDataSetIterator(
            data_generator=self,
            n_labeled=self.labeled_data_set.size,
            labeled_batch_size=self.num_labeled_per_batch,
            n_unlabeled=self.unlabeled_data_set.size if self.using_unlabeled_data else 0,
            unlabeled_batch_size=self.num_unlabeled_per_batch if self.using_unlabeled_data else 0,
            shuffle=self.shuffle_data_after_epoch,
            seed=self.random_seed,
            initial_epoch=self.initial_epoch)

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
               self.num_unlabeled_per_batch > 0 and \
               self.batch_data_format != BatchDataFormat.SUPERVISED

    def get_data_batch(self, step_idx, labeled_batch, unlabeled_batch):
        super(ClassificationDataGenerator, self).get_data_batch(step_idx, labeled_batch, unlabeled_batch)

        crop_shape = self.get_batch_crop_shape()
        resize_shape = self.get_batch_resize_shape()

        #self.logger.debug_log('Batch crop shape: {}, resize shape: {}'.format(crop_shape, resize_shape))
        #self.logger.debug_log('Generating batch data for step {}: labeled: {}, ul: {}'.format(step_idx, labeled_batch, unlabeled_batch))

        stime = time.time()
        X, Y, W = self.get_labeled_batch_data(step_idx, labeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        X_unlabeled, Y_unlabeled, W_unlabeled = self.get_unlabeled_batch_data(step_idx, unlabeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        self.logger.debug_log('Raw data generation took: {}s'.format(time.time() - stime))

        X = X + X_unlabeled
        Y = Y + Y_unlabeled
        W = W + W_unlabeled

        num_unlabeled_samples_in_batch = len(X_unlabeled)
        num_samples_in_batch = len(X)

        # Cast the lists to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)

        # Normalize the photo batch data: color values to [-1, 1], subtract per pixel mean and divide by stddev
        X = self._normalize_image_batch(X)
        X_teacher = None

        # Generate possible mean teacher data
        # Note: only applied to inputs ground truth must be the same
        if self.generate_mean_teacher_data:
            stime_t_data = time.time()
            X_teacher = self._get_mean_teacher_data_from_image_batch(X)
            self.logger.debug_log('Mean Teacher data generation took: {}s'.format(time.time()-stime_t_data))

        # Apply possible gaussian noise
        # Note: applied after generating mean teacher data so teacher can have unnoised data during generation
        if self.use_data_augmentation:
            if self.data_augmentation_params.using_gaussian_noise:
                gaussian_noise_stddev = self.data_augmentation_params.gaussian_noise_stddev_function(step_idx)
                X = X + np.random.normal(loc=0.0, scale=gaussian_noise_stddev, size=X.shape)

        if self.batch_data_format == BatchDataFormat.SUPERVISED:
            batch_input_data = [X]
            batch_output_data = [Y]
        elif self.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # The dimensions of the number of unlabeled in the batch must match with batch dimension
            num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

            # Generate a dummy output for the dummy loss function and yield a batch of data
            dummy_output = np.zeros(shape=[num_samples_in_batch], dtype=np.int32)

            batch_input_data = [X, Y, W, num_unlabeled]

            if X.shape[0] != Y.shape[0] or X.shape[0] != W.shape[0] or X.shape[0] != num_unlabeled.shape[0]:
                self.logger.warn('Unmatching input first (batch) dimensions: {}, {}, {}, {}'.format(X.shape[0], Y.shape[0], W.shape[0], num_unlabeled.shape[0]))

            logits_output = Y
            batch_output_data = [dummy_output, logits_output]
        else:
            raise ValueError('Unknown batch data format: {}'.format(self.batch_data_format))

        # Append the mean teacher data to the batch input data
        if self.generate_mean_teacher_data:
            if X_teacher is None:
                raise ValueError('Supposed to generate teacher data but X_teacher is None')

            batch_input_data.append(X_teacher)

        self.logger.debug_log('Data generation took in total: {}s'.format(time.time() - stime))

        # If we are in debug mode, save the batch images
        if settings.DEBUG:
            b_min = np.min(X)
            b_max = np.max(X)
            b_min_teacher = np.min(X_teacher) if X_teacher is not None else 0
            b_max_teacher = np.max(X_teacher) if X_teacher is not None else 0

            for i in range(0, len(X)):
                label = np.argmax(Y[i])
                img = ((X[i] - b_min) / (b_max - b_min)) * 255.0
                self.logger.debug_log_image(img, '{}_{}_{}_{}_photo.jpg'.format(label, self.name, step_idx, i), scale=False)

                if X_teacher is not None:
                    img = ((X_teacher[i] - b_min_teacher) / (b_max_teacher - b_min_teacher)) * 255.0
                    self.logger.debug_log_image(img, '{}_{}_{}_{}_photo_teacher.jpg'.format(label, self.name, step_idx, i), scale=False)

        return batch_input_data, batch_output_data

    def get_labeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):

        if self.labeled_data_set.data_set_type == MINCDataSetType.MINC_2500:
            data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
                (delayed(pickle_method)(self, 'get_labeled_sample_minc_2500', step_index=step_index, sample_index=sample_index, resize_shape=resize_shape) for sample_index in index_array)
        elif self.labeled_data_set.data_set_type == MINCDataSetType.MINC:
            data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
                (delayed(pickle_method)(self, 'get_labeled_sample_minc', step_index=step_index, sample_index=sample_index, crop_shape=crop_shape, resize_shape=resize_shape) for sample_index in index_array)
        else:
            raise ValueError('Unknown data set type: {}'.format(self.labeled_data_set.data_set_type))

        X, Y, W = zip(*data)

        return list(X), list(Y), list(W)

    def get_labeled_sample_minc_2500(self, step_index, sample_index, resize_shape):
        minc_sample = self.labeled_data_set.samples[sample_index]
        img_file = self.labeled_data_set.photo_image_set.get_image_file_by_file_name(minc_sample.file_name)
        pil_image = img_file.get_image(self.num_color_channels)
        np_image = img_to_array(pil_image)

        if img_file is None:
            raise ValueError('Could not find image from ImageSet with file name: {}'.format(minc_sample.file_name))

        # Check whether we need to resize the photo to a constant size
        if resize_shape is not None:
            np_image = self._resize_image(np_image, resize_shape=resize_shape, cval=self.photo_cval, interp='bicubic')

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            images, _ = self._apply_data_augmentation_to_images(images=[np_image],
                                                                cvals=[self.photo_cval],
                                                                interpolations=[ImageInterpolation.BICUBIC])

            # Unpack the photo
            np_image, = images

        # Make sure the image dimensions satisfy the div2_constraint
        np_image = self._fit_image_to_div2_constraint(np_image=np_image, cval=self.photo_cval, interp='bicubic')

        # Construct label vector (one-hot)
        custom_label = self.labeled_data_set.minc_label_to_custom_label[minc_sample.minc_label]
        y = np.zeros(self.labeled_data_set.num_classes, dtype=np.float32)
        y[custom_label] = 1.0

        # Construct weight vector
        w = self.class_weights

        return np_image, y, w

    def get_labeled_sample_minc(self, step_index, sample_index, crop_shape, resize_shape):
        minc_sample = self.labeled_data_set.samples[sample_index]
        img_file = self.labeled_data_set.photo_image_set.get_image_file_by_file_name(minc_sample.file_name)
        pil_image = img_file.get_image(self.num_color_channels)
        np_image = img_to_array(pil_image)

        if img_file is None:
            raise ValueError('Could not find image from ImageSet with file name: {}'.format(minc_sample.file_name))

        if crop_shape is None:
            raise ValueError('MINC data set images cannot be used without setting a crop shape')

        # Check whether we need to resize the photo to a constant size
        if resize_shape is not None:
            np_image = self._resize_image(np_image, resize_shape=resize_shape, cval=self.photo_cval, interp='bicubic')

        img_height, img_width = np_image.shape[0], np_image.shape[1]
        crop_height, crop_width = crop_shape[0], crop_shape[1]
        crop_center_y, crop_center_x = minc_sample.y, minc_sample.x

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            np_image_orig = np.array(np_image, copy=True)
            images, transform = self._apply_data_augmentation_to_images(images=[np_image],
                                                                        cvals=[self.photo_cval],
                                                                        interpolations=[ImageInterpolation.BICUBIC],
                                                                        transform_origin=np.array([crop_center_y, crop_center_x]))

            # Unpack the photo
            np_image, = images

            crop_center_new = transform.transform_normalized_coordinates(np.array([crop_center_x, crop_center_y]))
            crop_center_x_new, crop_center_y_new = crop_center_new[0], crop_center_new[1]

            # If the center has gone out of bounds abandon the augmentation - otherwise, update the crop center values
            if (not 0.0 <= crop_center_y_new <= 1.0) or (not 0.0 <= crop_center_x_new <= 1.0):
                np_image = np_image_orig
            else:
                crop_center_y = crop_center_y_new
                crop_center_x = crop_center_x_new

        # Crop the image with the specified crop center. Regions going out of bounds are padded with a
        # constant value.
        y_c = crop_center_y*img_height
        x_c = crop_center_x*img_width

        # MINC crop values can go out of bounds so we can keep the crop size constant
        y_0 = int(round(y_c - crop_height*0.5))
        x_0 = int(round(x_c - crop_width*0.5))
        y_1 = int(round(y_c + crop_height*0.5))
        x_1 = int(round(x_c + crop_width*0.5))

        if y_1 - y_0 != crop_shape[0]:
            add = abs(y_1 - y_0)
            y_1 += add

        if x_1 - x_0 != crop_shape[1]:
            add = abs(x_1 - x_0)
            x_1 += add

        np_image = image_utils.np_crop_image_with_fill(np_image, x1=x_0, y1=y_0, x2=x_1, y2=y_1, cval=self.photo_cval)
        np_image = self._fit_image_to_div2_constraint(np_image=np_image, cval=self.photo_cval, interp='bicubic')

        # Construct label vector (one-hot)
        custom_label = self.labeled_data_set.minc_label_to_custom_label[minc_sample.minc_label]
        y = np.zeros(self.labeled_data_set.num_classes, dtype=np.float32)
        y[custom_label] = 1.0

        # Construct class weight vector
        w = self.class_weights

        return np_image, y, w

    def get_unlabeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        if not self.using_unlabeled_data:
            return [], [], []

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        unlabeled_data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
            (delayed(pickle_method)
             (self, 'get_unlabeled_sample', step_index=step_index, sample_index=sample_index, crop_shape=crop_shape, resize_shape=resize_shape) for sample_index in index_array)

        X_unlabeled, Y_unlabeled, W_unlabeled = zip(*unlabeled_data)

        return list(X_unlabeled), list(Y_unlabeled), list(W_unlabeled)

    def get_unlabeled_sample(self, step_index, sample_index, crop_shape, resize_shape):
        img_file = self.unlabeled_data_set.get_index(sample_index)
        pil_image = img_file.get_image(self.num_color_channels)
        np_image = img_to_array(pil_image)

        # Check whether we need to resize the photo and the mask to a constant size
        if resize_shape is not None:
            np_image = self._resize_image(np_image, resize_shape=resize_shape, cval=self.photo_cval, interp='bicubic')

        # Check whether any of the image dimensions is smaller than the crop,
        # if so pad with the assigned fill colors
        if crop_shape is not None and (np_image.shape[0] < crop_shape[0] or np_image.shape[1] < crop_shape[1]):
            # Image dimensions must be at minimum the same as the crop dimension
            # on each axis. The photo needs to be filled with the photo_cval
            min_img_shape = (max(crop_shape[0], np_image.shape[0]), max(crop_shape[1], np_image.shape[1]))
            np_image = image_utils.np_pad_image_to_shape(np_image, min_img_shape, self.photo_cval)

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            images, _ = self._apply_data_augmentation_to_images(images=[np_image],
                                                                cvals=[self.photo_cval],
                                                                interpolations=[ImageInterpolation.BICUBIC])
            # Unpack the photo
            np_image, = images

        # If a crop size is given: take a random crop of the image
        if crop_shape is not None:
            x1y1, x2y2 = image_utils.np_get_random_crop_area(np_image, crop_shape[1], crop_shape[0])
            np_image = image_utils.np_crop_image(np_image, x1y1[0], x1y1[1], x2y2[0], x2y2[1])

        np_image = self._fit_image_to_div2_constraint(np_image=np_image, cval=self.photo_cval, interp='bicubic')

        # Create a dummy label vector (one-hot) all zeros
        y = self.dummy_label_vector

        # Construct class weight vector
        w = self.class_weights

        return np_image, y, w
