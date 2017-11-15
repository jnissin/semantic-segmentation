# coding=utf-8

import time
import random
import os
import numpy as np

from enum import Enum
from abc import ABCMeta, abstractmethod, abstractproperty

from PIL import Image as PImage
from PIL.Image import Image as PILImage
from PIL.ImageFile import ImageFile as PILImageFile

from utils import dataset_utils
from utils import image_utils
from utils.image_utils import ImageInterpolationType, ImageTransform, img_to_array
from utils.dataset_utils import MaterialClassInformation, MaterialSample, MINCSample, BoundingBox
from data_set import LabeledImageDataSet, UnlabeledImageDataSet, ImageFile, ImageSet
from iterators import DataSetIterator, BasicDataSetIterator, MaterialSampleDataSetIterator
from logger import Logger
from enums import BatchDataFormat, SuperpixelSegmentationFunctionType, ImageType, MaterialSampleIterationMode

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
                 batch_data_format,
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
                 generate_mean_teacher_data=False,
                 resized_image_cache_path=None):
        """
        Builds a wrapper for DataGenerator parameters

        # Arguments
            :param batch_data_format: semi-supervised or supervised data format
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
        # type: (BatchDataFormat, int, str, int, list, list, bool, np.ndarray, bool, np.ndarray, np.ndarray, bool, DataAugmentationParameters, bool, int, int, bool, str) -> None

        self.batch_data_format = batch_data_format
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
        self.resized_image_cache_path = resized_image_cache_path


class SegmentationDataGeneratorParameters(DataGeneratorParameters):
    """
    Helps to maintain parameters of the segmentation data generator
    """

    def __init__(self,
                 material_class_information,
                 mask_cval=None,
                 use_material_samples=False,
                 material_sample_iteration_mode=MaterialSampleIterationMode.NONE,
                 use_selective_attention=False,
                 use_adaptive_sampling=False,
                 num_crop_reattempts=0,
                 superpixel_segmentation_function=None,
                 superpixel_mask_cache_path=None,
                 **kwargs):
        # type: (list[MaterialClassInformation], np.ndarray, bool, MaterialSampleIterationMode, bool, bool, int, SuperpixelSegmentationFunctionType, str) -> None

        """
        Builds a wrapper for SegmentationDataGenerator parameters

        # Arguments
            :param material_class_information: material class information list
            :param mask_cval: fill color value for masks [0,255], otherwise zeros matching mask encoding are used
            :param use_material_samples: should material samples be used
            :param material_sample_iteration_mode: iteration mode for material samples
            :param use_selective_attention: should we use selective attention (mark everything else as bg besides the material sample material)
            :param use_adaptive_sampling: should we use adaptive sampling (adapt sampling probability according to pixels seen per category)
            :param superpixel_segmentation_function: which function to use to generate superpixel segmentation for unlabeled samples
            :param superpixel_mask_cache_path: path for storing cached superpixel segmentations
        # Returns
            Nothing
        """
        super(SegmentationDataGeneratorParameters, self).__init__(**kwargs)

        self.material_class_information = material_class_information
        self.mask_cval = mask_cval
        self.use_material_samples = use_material_samples
        self.material_sample_iteration_mode = material_sample_iteration_mode
        self.use_selective_attention = use_selective_attention
        self.use_adaptive_sampling = use_adaptive_sampling
        self.num_crop_reattempts = num_crop_reattempts
        self.superpixel_segmentation_function = superpixel_segmentation_function
        self.superpixel_mask_cache_path = superpixel_mask_cache_path


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

    def __init__(self, params):
        # type: (DataGeneratorParameters) -> None

        # UUID might be before the init function is called
        if not hasattr(self, '_uuid'):
            self._uuid = None

        self._logger = None

        self.logger.log('DataGenerator: Initializing data generator with UUID: {}'.format(self.uuid))

        # Unwrap DataGeneratorParameters to member variables
        self.batch_data_format = params.batch_data_format
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
        self.resized_image_cache_path = params.resized_image_cache_path

        # If caching resized images ensure the cache path exists
        if self.resized_image_cache_path is not None and not os.path.exists(self.resized_image_cache_path):
            self.logger.log('Creating resized image cache to: {}'.format(self.resized_image_cache_path))
            os.makedirs(os.path.dirname(self.resized_image_cache_path))

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
            self.per_channel_mean = np.array(dataset_utils.calculate_per_channel_mean(self.get_all_photos()))
            self.logger.log('DataGenerator: Using per-channel mean: {}'.format(list(self.per_channel_mean)))

        # Calculate missing per-channel stddev if necessary
        if self.use_per_channel_stddev_normalization and (self.per_channel_stddev is None or len(self.per_channel_stddev) != self.num_color_channels):
            self.per_channel_stddev = np.array(dataset_utils.calculate_per_channel_stddev(self.get_all_photos(), self.per_channel_mean))
            self.logger.log('DataGenerator: Using per-channel stddev: {}'.format(list(self.per_channel_stddev)))

        # Use per-channel mean but in range [0, 255] if nothing else is given.
        # The normalization is done to the whole batch after transformations so
        # the images are not in range [-1,1] before transformations.
        if self.photo_cval is None:
            if self.per_channel_mean is None:
                self.photo_cval = np.zeros(3, dtype=np.int32)
            else:
                self.photo_cval = np.round(image_utils.np_from_normalized_to_255(self.per_channel_mean)).astype(np.int32)
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

    def _pil_resize_image(self, img, resize_shape, cval, interp, img_type=ImageType.NONE):
        # type: (PILImage, tuple, np.ndarray, ImageInterpolationType, ImageType) -> PILImage

        # If the resize shape is None just return the original image
        if resize_shape is None:
            return img

        assert(isinstance(resize_shape, list) or isinstance(resize_shape, tuple) or isinstance(resize_shape, np.ndarray))
        assert(len(resize_shape) == 2)
        assert(not (resize_shape[0] is None and resize_shape[1] is None))

        # Scale to match the sdim found in index 0
        if resize_shape[0] is not None and resize_shape[1] is None:
            target_sdim = resize_shape[0]
            img_sdim = min(img.size)

            if target_sdim == img_sdim:
                return img

            scale_factor = float(target_sdim)/float(img_sdim)
            target_shape = (int(round(img.height * scale_factor)), int(round(img.width * scale_factor)))
        # Scale the match the bdim found in index 1
        elif resize_shape[0] is None and resize_shape[1] is not None:
            target_bdim = resize_shape[1]
            img_bdim = max(img.size)

            if target_bdim == img_bdim:
                return img

            scale_factor = float(target_bdim)/float(img_bdim)
            target_shape = (int(round(img.height * scale_factor)), int(round(img.width * scale_factor)))
        # Scale to the exact shape
        else:
            target_shape = tuple(resize_shape)

            if target_shape[0] == img.height and target_shape[1] == img.width:
                return img

        # If using caching
        if self.resized_image_cache_path is not None and isinstance(img, PILImageFile):
            # Cached file name is: <file_name>_<height>_<width>_<interp>_<img_type><file_ext>
            cached_img_name = os.path.splitext(os.path.basename(img.filename))
            file_ext = cached_img_name[1]
            cached_img_name = '{}_{}_{}_{}_{}{}'.format(cached_img_name[0],
                                                        target_shape[0],
                                                        target_shape[1],
                                                        interp.value,
                                                        img_type.value,
                                                        file_ext)
            cached_img_path = os.path.join(self.resized_image_cache_path, cached_img_name)

            # If the cached file exists load and return it
            if os.path.exists(cached_img_path):
                try:
                    resized_img = image_utils.load_img(cached_img_path, num_read_attemps=2)
                    return resized_img
                # Log the exception and default to the non cached resize
                except Exception as e:
                    self.logger.warn('Caught exception during resized image caching (read): {}'.format(e.message))

            # If there was no cached file - resize using PIL and cache
            # Use the same save format as the original file if it is given
            save_format = img.format

            if save_format is None:
                # Try determining format from file ending
                if file_ext.lower() == '.jpg' or file_ext.lower() == '.jpeg':
                    save_format = 'JPEG'
                elif file_ext.lower() == '.png':
                    save_format = 'PNG'
                else:
                    # If we couldn't determine format from extension use the img type to guess
                    save_format = 'PNG' if img_type.MASK else 'JPEG'

            try:
                resized_img = image_utils.pil_resize_image_with_padding(img, shape=target_shape, cval=cval, interp=interp)
                resized_img.save(cached_img_path, format=save_format)
                return resized_img
            except Exception as e:
                self.logger.warn('Caught exception during resized image caching (write): {}'.format(e.message))

        # If everything else fails - just resize to target shape without caching
        return image_utils.pil_resize_image_with_padding(img, shape=target_shape, cval=cval, interp=interp)

    def _pil_fit_image_to_div2_constraint(self, img, cval, interp):
        # type: (PILImage, np.ndarray, ImageInterpolationType) -> PILImage

        # Make sure the image dimensions satisfy the div2_constraint i.e. are n times divisible
        # by 2 to work with the network. If the dimensions are not ok pad the images.
        img_height_div2 = dataset_utils.count_trailing_zeroes(img.height)
        img_width_div2 = dataset_utils.count_trailing_zeroes(img.width)

        if img_height_div2 < self.div2_constraint or img_width_div2 < self.div2_constraint:
            target_shape = dataset_utils.get_required_image_dimensions((img.height, img.width), self.div2_constraint)
            img = image_utils.pil_resize_image_with_padding(img, shape=target_shape, cval=cval, interp=interp)

        return img

    def _np_normalize_image_batch(self, batch):
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
        batch /= 255.0
        batch -= 0.5
        batch *= 2.0

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

    def _get_random_crop_area(self, img_width, img_height, crop_width, crop_height):
        # type: (int, int, int, int) -> tuple[tuple[int, int], tuple[int, int]]

        """
        The function returns a random crop from the image as (y1, x1), (y2, x2).

        # Arguments
            :param img_width: image width
            :param img_height: image height
            :param crop_width: width of the crop
            :param crop_height: height of the crop

        # Returns
            :return: two integer tuples describing the crop: (y1, x1), (y2, x2)
        """

        if crop_width > img_width or crop_height > img_height:
            raise ValueError('Crop dimensions exceed the image dimensions: [{},{}] vs [{},{}]'.format(crop_height, crop_width, img_height, img_width))

        x1 = np.random.randint(0, img_width - crop_width + 1)
        y1 = np.random.randint(0, img_height - crop_height + 1)
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        return (y1, x1), (y2, x2)

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

    def _pil_apply_data_augmentation_to_images(self, images, cvals, random_seed, interpolations, transform_origin=None, override_channel_shift_ranges=None, override_gamma_adjust_ranges=None):
        # type: (list[PILImage], list, int, list, np.ndarray, list, list) -> (list[PILImage], ImageTransform)

        """
        Applies (same) data augmentation as per stored DataAugmentationParameters to a list of images.

        # Arguments
            :param images: images to augment (same augmentation on all)
            :param cvals: constant fill values for the images
            :param random_seed: random seed
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
        images, transform = image_utils.pil_apply_random_image_transform(images=images,
                                                                         cvals=cvals,
                                                                         random_seed=random_seed,
                                                                         interpolations=interpolations,
                                                                         transform_origin=transform_origin,
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

    def _get_mean_teacher_data_from_image_batch(self, X, dtype=np.float32):
        # type: (list[PILImage], dtype) -> np.ndarray

        if not self.generate_mean_teacher_data:
            raise ValueError('Request to get mean teacher data when generate_mean_teacher_data is False')

        # If not applying any noise - return the same images
        if not self.data_augmentation_params.using_mean_teacher_noise:
            X_teacher = [img_to_array(img) for img in X]
            X_teacher = np.asarray(X_teacher, dtype=dtype)
            return X_teacher

        # Apply noise transformations individually to each image
        # * Horizontal flips
        # * Vertical flips
        # * Translations
        # * Intensity shifts
        # * Gaussian noise
        noise_params = self.data_augmentation_params.mean_teacher_noise_params

        X_teacher = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')(
            delayed(pickle_method)(self, '_apply_mean_teacher_noise_to_image', img=img) for img in X)

        # Transform from PIL to Numpy
        X_teacher = [img_to_array(img) for img in X_teacher]
        X_teacher = np.asarray(X_teacher, dtype=dtype)

        # Normalize
        X_teacher = self._np_normalize_image_batch(X_teacher)

        # Apply gaussian noise to the whole batch at once
        gaussian_noise_stddev = noise_params.get('gaussian_noise_stddev')

        if gaussian_noise_stddev is not None:
            X_teacher += np.random.normal(loc=0.0, scale=gaussian_noise_stddev, size=X_teacher.shape)

        return X_teacher

    def _apply_mean_teacher_noise_to_image(self, img):
        # type: (PILImage) -> PILImage

        noise_params = self.data_augmentation_params.mean_teacher_noise_params
        teacher_img = img.copy()

        # Apply brightness shift
        translate_range = noise_params.get('translate_range')
        rotation_range = noise_params.get('rotation_range')
        horizontal_flip_probability = noise_params.get('horizontal_flip_probability')
        vertical_flip_probability = noise_params.get('vertical_flip_probability')
        channel_shift_range = noise_params.get('channel_shift_range')
        gamma_adjust_range = noise_params.get('gamma_adjust_range')

        # Apply gamma adjust
        if gamma_adjust_range is not None:
            gamma = np.random.uniform(1.0 - gamma_adjust_range, 1.0 + gamma_adjust_range)
            teacher_img = image_utils.pil_adjust_gamma(teacher_img, gamma)

        # Apply channel shifts
        if channel_shift_range is not None:
            intensity = int(round(np.random.uniform(-channel_shift_range, channel_shift_range) * 255.0))
            teacher_img = image_utils.pil_intensity_shift(teacher_img, intensity=intensity)

        # Apply horizontal flips
        if horizontal_flip_probability is not None:
            if np.random.random() < horizontal_flip_probability:
                teacher_img = image_utils.pil_apply_flip(teacher_img, method=PImage.FLIP_LEFT_RIGHT)

        # Apply vertical flips
        if vertical_flip_probability is not None:
            if np.random.random() < vertical_flip_probability:
                teacher_img = image_utils.pil_apply_flip(teacher_img, method=PImage.FLIP_TOP_BOTTOM)

        # Spatial transforms
        offset = (teacher_img.width * 0.5, teacher_img.height * 0.5)
        translate_x = np.random.uniform(-translate_range, translate_range) * teacher_img.width if translate_range is not None else 0.0
        translate_y = np.random.uniform(-translate_range, translate_range) * teacher_img.height if translate_range is not None else 0.0
        theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range)) if rotation_range is not None else 0.0
        scale = 1.0

        transform = image_utils.pil_create_transform(offset=offset, translate=(translate_x, translate_y), theta=theta, scale=scale)
        teacher_img = image_utils.pil_transform_image(teacher_img, transform=transform, resample=image_utils.ImageInterpolationType.BICUBIC.value, cval=self.photo_cval)

        return teacher_img


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
                 params,
                 class_weights=None):
        # type: (LabeledImageDataSet, UnlabeledImageDataSet, int, int, SegmentationDataGeneratorParameters, np.ndarray) -> None

        """
        # Arguments
            :param labeled_data_set: LabeledImageSet instance of the labeled data set
            :param unlabeled_data_set: UnlabeledImageSet instance of the unlabeled data set
            :param num_labeled_per_batch: number of labeled images per batch
            :param num_unlabeled_per_batch: number of unlabeled images per batch
            :param params: SegmentationDataGeneratorParameters object
            :param class_weights: class weights
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
        self.superpixel_segmentation_function = params.superpixel_segmentation_function
        self.superpixel_mask_cache_path = params.superpixel_mask_cache_path
        self.material_sample_iteration_mode = params.material_sample_iteration_mode

        super(SegmentationDataGenerator, self).__init__(params)

        # Create the superpixel cache path if it doesn't exist
        if self.superpixel_mask_cache_path is not None and not os.path.exists(self.superpixel_mask_cache_path):
            self.logger.log('Creating superpixel mask cache to: {}'.format(self.superpixel_mask_cache_path))
            os.makedirs(os.path.dirname(self.superpixel_mask_cache_path))

        # Build a look up dictionary material red color -> class idx
        self.material_r_color_to_material_class = {}

        for material in self.material_class_information:
            for r_color in material.r_color_values:
                # Paranoid check for duplicate red color values
                if r_color in self.material_r_color_to_material_class:
                    raise ValueError('Material red color already in lookup dictionary r_color: {}, existing mapping: {} -> {}'
                                     .format(r_color, r_color, self.material_r_color_to_material_class.get(r_color)))

                self.material_r_color_to_material_class[r_color] = material.id


        # Use black (background)
        if self.mask_cval is None:
            self.mask_cval = np.zeros(3, dtype=np.int32)
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

        self.logger.log('Use material samples: {}, material sample iteration mode: {}'.format(self.use_material_samples, self.material_sample_iteration_mode))
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
                iteration_mode=self.material_sample_iteration_mode,
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
            :return: labeled data as two lists: (X, Y)
        """

        if self.use_material_samples:
            labeled_batch_files, material_samples = self.labeled_data_set.get_files_and_material_samples(index_array)
        else:
            labeled_batch_files, material_samples = self.labeled_data_set.get_indices(index_array), None

        # Process the labeled files for this batch
        labeled_data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
            (delayed(pickle_method)
             (self,
              'get_labeled_segmentation_data_pair',
              step_idx=step_index,
              photo_file=labeled_batch_files[i][0],
              mask_file=labeled_batch_files[i][1],
              material_sample=material_samples[i] if material_samples is not None else None,
              crop_shape=crop_shape,
              resize_shape=resize_shape) for i in range(len(labeled_batch_files)))

        # Unzip the photo mask pairs
        X, Y = zip(*labeled_data)
        return X, Y

    def get_unlabeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        # type: (int, np.array[int], list, list) -> (list[np.array], list[np.array])

        """
        # Arguments
            :param step_index: index of the current step
            :param index_array: indices of the unlabeled data
            :param crop_shape: crop shape or None
            :param resize_shape: resize shape or None
        # Returns
            :return: unlabeled data as two lists: (X, Y)
        """

        # If we don't have unlabeled data return two empty lists
        if not self.using_unlabeled_data:
            return (), ()

        unlabeled_batch_files = self.unlabeled_data_set.get_indices(index_array)

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        unlabeled_data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
            (delayed(pickle_method)
             (self,
              'get_unlabeled_segmentation_data_pair',
              step_idx=step_index,
              photo_file=photo_file,
              crop_shape=crop_shape,
              resize_shape=resize_shape) for photo_file in unlabeled_batch_files)

        X_unlabeled, Y_unlabeled = zip(*unlabeled_data)
        return X_unlabeled, Y_unlabeled

    def get_labeled_segmentation_data_pair(self, step_idx, photo_file, mask_file, material_sample, crop_shape, resize_shape):
        # type: (int, ImageFile, ImageFile, MaterialSample, list, list) -> (PILImage, PILImage)

        """
        Returns a photo mask pair for supervised segmentation training. Will apply data augmentation
        and cropping as instructed in the parameters.

        The photos are not normalized to range [-1,1] within the function.

        # Arguments
            :param step_idx: index of the current training step
            :param photo_file: photo as ImageFile
            :param mask_file: segmentation mask as ImageFile
            :param material_sample: material sample information for the files
        # Returns
            :return: a tuple of numpy arrays (image, mask)
        """

        # Load the image and mask as PIL images
        # Note: discard everything but the red band in the mask image since we are not
        # using anything else
        pil_photo = photo_file.get_image(color_channels=self.num_color_channels)
        pil_mask = mask_file.get_image(color_channels=self.num_color_channels)

        # Resize the photo to match the mask size if necessary, since
        # the original photos are sometimes huge
        if pil_photo.size != pil_mask.size:
            pil_photo = pil_photo.resize(pil_mask.size, PImage.BICUBIC)

        if pil_photo.size != pil_mask.size:
            raise ValueError('Non-matching photo and mask dimensions after resize: {} != {}'.format(pil_photo.size, pil_mask.size))

        # Apply crops and augmentation
        pil_photo, pil_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx,
                                                                        pil_photo=pil_photo,
                                                                        pil_mask=pil_mask,
                                                                        photo_cval=self.photo_cval,
                                                                        mask_cval=self.mask_cval,
                                                                        material_sample=material_sample,
                                                                        crop_shape=crop_shape,
                                                                        resize_shape=resize_shape,
                                                                        validate_crops=True)

        if material_sample is not None:
            # Sanity check: material samples are supposed to guarantee material instances
            if not self._mask_crop_is_valid(pil_mask, requested_material_r_color=material_sample.material_r_color):
                photo_filename = '{}_crop_missing_id_{}_r_{}.jpg'.format(photo_file.file_name, material_sample.material_id, material_sample.material_r_color)
                mask_filename = '{}_crop_missing_id_{}_r_{}.png'.format(photo_file.file_name, material_sample.material_id, material_sample.material_r_color)

                self.logger.log_image(pil_photo, file_name=photo_filename, format='JPEG', scale=False)
                self.logger.log_image(pil_mask, file_name=mask_filename, format='PNG', scale=False)

                unique_red_colors = image_utils.pil_image_get_unique_band_values(pil_mask, band=0)
                self.logger.warn('Crop of material sample with material id: {} was missing red color: {}. Found red colors: {}'
                                 .format(material_sample.material_id, material_sample.material_r_color, unique_red_colors))

            # If we are using selective attention mark everything else as zero (background) besides
            # the red channel representing the current material sample
            if self.use_selective_attention:
                pil_mask = image_utils.pil_image_mask_by_band_value(pil_mask, band=0, val=material_sample.material_r_color, cval=0)

        return pil_photo, pil_mask

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

        # Load the image
        pil_photo = photo_file.get_image(color_channels=self.num_color_channels)

        # Generate mask for the photo - note: the labels are generated before cropping
        # and augmentation to capture global structure within the image
        pil_mask = self._generate_mask_for_unlabeled_image(pil_photo)

        # Apply crops and augmentation
        pil_photo, pil_mask = self.process_segmentation_photo_mask_pair(step_idx=step_idx,
                                                                        pil_photo=pil_photo,
                                                                        pil_mask=pil_mask,
                                                                        photo_cval=self.photo_cval,
                                                                        mask_cval=[0],
                                                                        crop_shape=crop_shape,
                                                                        resize_shape=resize_shape,
                                                                        validate_crops=False,
                                                                        dummy_mask=self.superpixel_segmentation_function is SuperpixelSegmentationFunctionType.NONE)

        return pil_photo, pil_mask

    def process_segmentation_photo_mask_pair(self, step_idx, pil_photo, pil_mask, photo_cval, mask_cval, material_sample=None, crop_shape=None, resize_shape=None, validate_crops=True, dummy_mask=False):
        # type: (int, PILImage, PILImage, np.ndarray, np.ndarray, MaterialSample, tuple, tuple, bool) -> (PILImage, PILImage)

        """
        Applies crop and data augmentation to two numpy arrays representing the photo and
        the respective segmentation mask. The photos are not normalized to range [-1,1]
        within the function.

        # Arguments
            :param step_idx: index of the current step
            :param pil_photo: the photo as a PIL image
            :param pil_mask: the mask as a PIL image, must have matching dimensions with the pil_photo
            :param photo_cval: photo fill value in range [0, 255]
            :param mask_cval: mask fill value in range [0, 255]
            :param material_sample: the material sample
            :param crop_shape: shape of the crop or None
            :param resize_shape: shape of the target resize or None
            :param validate_crops: retries crops if the whole crop is 0 (BG)
            :param dummy_mask: is the mask a dummy mask i.e. all black (allows skipping augmentation for masks)
        # Returns
            :return: a tuple of (photo, mask) as PIL images
        """

        if pil_photo.size != pil_mask.size:
            raise ValueError('Non-matching photo and mask sizes: {} != {}'.format(pil_photo.size, pil_mask.size))

        if material_sample is not None and crop_shape is None:
            raise ValueError('Cannot use material samples without cropping')

        # Check whether we need to resize the photo and the mask to a constant size
        if resize_shape is not None:
            pil_photo = self._pil_resize_image(pil_photo, resize_shape=resize_shape, cval=photo_cval, interp=ImageInterpolationType.BICUBIC, img_type=ImageType.PHOTO)
            pil_mask = self._pil_resize_image(pil_mask, resize_shape=resize_shape, cval=mask_cval, interp=ImageInterpolationType.NEAREST, img_type=ImageType.MASK)

        # Drop any unnecessary channels from the mask image we only use the red or L (luma for grayscale)
        if len(pil_mask.getbands()) > 1:
            pil_mask = pil_mask.split()[0]

        # Ensure the cval matches the new representation (single channel)
        if mask_cval is not None:
            mask_cval = mask_cval[0:1]

        # Check whether any of the image dimensions is smaller than the crop,
        # if so pad with the assigned fill colors
        if crop_shape is not None and (pil_photo.height < crop_shape[0] or pil_photo.width < crop_shape[1]):
            # Image dimensions must be at minimum the same as the crop dimension
            # on each axis. The photo needs to be filled with the photo_cval color and masks
            # with the mask cval color
            min_img_shape = (max(crop_shape[0], pil_photo.height), max(crop_shape[1], pil_photo.width))
            pil_photo = image_utils.pil_pad_image_to_shape(pil_photo, min_img_shape, photo_cval)
            pil_mask = image_utils.pil_pad_image_to_shape(pil_mask, min_img_shape, mask_cval)

        # If we are using data augmentation apply the random transformation
        # to both the image and mask. The transformation is applied to the
        # whole image to decrease the number of 'dead' pixels due to transformations
        # within the possible crop only.
        bbox = material_sample.get_bbox_rel() if material_sample is not None else None

        if self._should_apply_augmentation(step_idx):
            pil_photo_orig = None
            pil_mask_orig = None

            if material_sample is not None:
                pil_photo_orig = pil_photo.copy()
                pil_mask_orig = pil_mask.copy() if not dummy_mask else None

            # If we have a non-dummy mask apply augmentation to both photo and mask, otherwise skip augmentation for the mask
            if not dummy_mask:
                images, transform = self._pil_apply_data_augmentation_to_images(images=[pil_photo, pil_mask],
                                                                                cvals=[photo_cval, mask_cval],
                                                                                random_seed=self.random_seed + step_idx,
                                                                                interpolations=[ImageInterpolationType.BICUBIC, ImageInterpolationType.NEAREST],
                                                                                override_channel_shift_ranges=[self.data_augmentation_params.channel_shift_range, None],
                                                                                override_gamma_adjust_ranges=[self.data_augmentation_params.gamma_adjust_range, None])
                # Unpack the images
                pil_photo, pil_mask = images
            else:
                images, transform = self._pil_apply_data_augmentation_to_images(images=[pil_photo],
                                                                                cvals=[photo_cval],
                                                                                random_seed=self.random_seed + step_idx,
                                                                                interpolations=[ImageInterpolationType.BICUBIC],
                                                                                override_channel_shift_ranges=[self.data_augmentation_params.channel_shift_range],
                                                                                override_gamma_adjust_ranges=[self.data_augmentation_params.gamma_adjust_range])
                # Unpack the image
                pil_photo, = images

            if material_sample is not None:
                # Check if the material was lost during transform
                material_lost_during_transform = not image_utils.pil_image_band_contains_value(pil_mask, 0, material_sample.material_r_color)

                if not material_lost_during_transform:
                    # Applies the same transform the bounding box coordinates
                    bbox = self._transform_bbox(bbox, transform, pil_mask, material_sample)

                # If the material was lost during the random transform - revert to unaugmented values or
                # If we cannot build a new valid axis-aligned bounding box - revert to unaugmented values
                if material_lost_during_transform or bbox is None:
                    self.logger.debug_log('Reverting to original bbox and image data, material lost: {}, cannot rebuild axis-aligned bbox: {}'
                                          .format(material_lost_during_transform, bbox is None))

                    bbox = material_sample.get_bbox_rel()
                    pil_photo = pil_photo_orig
                    pil_mask = pil_mask_orig
                else:
                    # Destroy the backup images
                    del pil_photo_orig
                    del pil_mask_orig

        # If a crop size is given: take a random crop of both the image and the mask
        if crop_shape is not None:
            for attempt in xrange(max(self.num_crop_reattempts+1, 1), 0, -1):
                if material_sample is None:
                    # If we don't have a bounding box as a hint for cropping - take random crops
                    y1x1, y2x2 = self._get_random_crop_area(img_width=pil_mask.width,
                                                            img_height=pil_mask.height,
                                                            crop_width=crop_shape[1],
                                                            crop_height=crop_shape[0])
                else:
                    # Use the bounding box information to take a targeted crop
                    y1x1, y2x2 = self._get_random_bbox_crop_area(bbox=bbox,
                                                                 img_width=pil_mask.width,
                                                                 img_height=pil_mask.height,
                                                                 crop_width=crop_shape[1],
                                                                 crop_height=crop_shape[0],
                                                                 material_sample=material_sample)

                pil_mask_crop = image_utils.pil_crop_image(pil_mask, x1=y1x1[1], y1=y1x1[0], x2=y2x2[1], y2=y2x2[0])

                # If we validate crops - check that the crop is non-black/has the material color
                if validate_crops and not dummy_mask:
                    if material_sample is not None:
                        valid_crop_found = self._mask_crop_is_valid(pil_mask_crop, requested_material_r_color=material_sample.material_r_color)
                    else:
                        valid_crop_found = self._mask_crop_is_valid(pil_mask_crop)
                else:
                    valid_crop_found = True

                # If a valid crop was found or this is the last attempt or we should not retry crops
                stop_iteration = valid_crop_found or attempt-1 <= 0 or not validate_crops

                if stop_iteration:
                    # If valid crop was not found at all after all the retry attempts
                    if not valid_crop_found:
                        if material_sample is not None and bbox is not None:
                            mask_unique_material_colors = image_utils.pil_image_get_unique_band_values(pil_mask, band=0)
                            crop_unique_material_colors = image_utils.pil_image_get_unique_band_values(pil_mask_crop, band=0)

                            self.logger.warn('Material not found within crop area of shape: {} for material id: {} and material red color: {}. '
                                             'Crop: {}, bbox: {}, img_size: {}. Mask image unique material colors: {}. Mask crop unique material colors: {}'
                                             .format(crop_shape, material_sample.material_id, material_sample.material_r_color,
                                                     (y1x1, y2x2), (bbox.top_left, bbox.bottom_right), (pil_mask.size[1], pil_mask.size[0]),
                                                     mask_unique_material_colors, crop_unique_material_colors))
                        else:
                            self.logger.warn('Only background found within crop area of shape: {}'.format(crop_shape))

                    pil_mask_crop.load()
                    pil_mask = pil_mask_crop
                    pil_photo = image_utils.pil_crop_image(pil_photo, x1=y1x1[1], y1=y1x1[0], x2=y2x2[1], y2=y2x2[0], load=True)
                    break

        # Make sure both photo and mask satisfy the div2 constraint
        pil_photo = self._pil_fit_image_to_div2_constraint(img=pil_photo, cval=photo_cval, interp=ImageInterpolationType.BICUBIC)
        pil_mask = self._pil_fit_image_to_div2_constraint(img=pil_mask, cval=mask_cval, interp=ImageInterpolationType.NEAREST)

        return pil_photo, pil_mask

    def _mask_crop_is_valid(self, pil_mask_crop, requested_material_r_color=None):
        # type: (PILImage, int) -> bool

        """
        Returns true if the material crop is valid i.e. contains the requested material red color
        or is not all black if no color is given.

        # Arguments
            :param pil_mask_crop: the crop area
            :param requested_material_r_color: requested material color
        # Returns
            :return: True if the crop is valid False otherwise
        """

        # If there is no requested color make sure the mask is not all black
        if requested_material_r_color is None:
            is_valid_crop = not image_utils.pil_image_band_only_contains_value(pil_mask_crop, band=0, val=0)
        else:
            is_valid_crop = image_utils.pil_image_band_contains_value(pil_mask_crop, band=0, val=requested_material_r_color)

        return is_valid_crop

    def _get_random_bbox_crop_area(self, bbox, img_width, img_height, crop_width, crop_height, material_sample):
        # type: (BoundingBox, int, int, int, int, MaterialSample) -> tuple[tuple[int, int], tuple[int, int]]

        """
        Generates a random crop area either containing the whole bounding box or within the bounding box
        depending on which area is bigger the crop area or the bounding box area.

        # Arguments
            :param bbox: the bounding box
            :param img_width: image width
            :param img_height: image height
            :param crop_width: crop width
            :param crop_height: crop height
            :param material_sample: the material sample
        # Return
            :return: The crop as (y1, x1), (y2, x2)
        """

        # Transform bbox to absolute coordinates from relative
        bbox_ymin = int(round(bbox.y_min * img_height))
        bbox_ymax = min(int(round(bbox.y_max * img_height)) + 1, img_height)    # +1 because the bbox ymax is inclusive
        bbox_xmin = int(np.round(bbox.x_min * img_width))
        bbox_xmax = min(int(np.round(bbox.x_max * img_width)) + 1, img_width)   # +1 because the bbox xmax is inclusive
        bbox_height = bbox_ymax - bbox_ymin
        bbox_width = bbox_xmax - bbox_xmin

        # Slack parameters allow the crop to go outside the bbox boundaries
        # by a given amount in case bbox > crop
        slack_coefficient = 0.1                                                 # Amount of slack if bbox dim > crop dim (%) of crop width/height
        slack_height = int(crop_height * slack_coefficient)
        slack_width = int(crop_width * slack_coefficient)

        # Apparently the material sample data can have single pixel height/width bounding boxes
        # expand the bounding box a bit to avoid errors
        if bbox_height <= 0:
            bbox_ymin = max(bbox_ymin - 1, 0)
            bbox_ymax = min(bbox_ymax + 1, img_height)
            bbox_height = bbox_ymax - bbox_ymin

        if bbox_width <= 0:
            bbox_xmin = max(bbox_xmin - 1, 0)
            bbox_xmax = min(bbox_xmax + 1, img_width)
            bbox_width = bbox_xmax - bbox_xmin

        # If after the fix bounding box is still of invalid size throw an error
        if bbox_height <= 0 or bbox_width <= 0:
            raise ValueError('Invalid bounding box dimensions: mat id: {}, file name: {}, bbox: {}, original bbox: {}'
                             .format(material_sample.material_id, material_sample.file_name, bbox.corners, material_sample.get_bbox_rel().corners))

        # Calculate the difference in height and width between the bbox and crop
        height_diff = abs(crop_height - bbox_height)
        width_diff = abs(crop_width - bbox_width)

        if bbox_height <= crop_height:
            crop_ymin = bbox_ymin - np.random.randint(0, min(height_diff + 1, bbox_ymin + 1))
        else:
            # If the crop is less tall than the bbox -> allow slack to show borders
            top_slack = max(-bbox_ymin, -slack_height)
            bottom_slack = slack_height
            crop_ymin = bbox_ymin + np.random.randint(top_slack, height_diff + 1 + bottom_slack)

        if bbox_width <= crop_width:
            crop_xmin = bbox_xmin - np.random.randint(0, min(width_diff + 1, bbox_xmin + 1))
        else:
            # If the crop is less wide than the bbox -> allow slack to show borders
            left_slack = max(-bbox_xmin, -slack_width)
            right_slack = slack_width
            crop_xmin = bbox_xmin + np.random.randint(left_slack, width_diff + 1 + right_slack)

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

        return (crop_ymin, crop_xmin), (crop_ymax, crop_xmax)

    def _transform_bbox(self, bbox, transform, pil_mask, material_sample):
        # type: (BoundingBox, ImageTransform, PILImage, MaterialSample) -> BoundingBox

        """
        Applies the image transform (transform) to the bounding box (bbox). Ensures that the transformed
        bounding box area contains the desired material sample red color, if not returns None.

        # Arguments
            :param bbox: the original bounding box
            :param transform: the transform used on the image
            :param pil_mask: the transformed image
            :param material_sample: the material sample that should be found within the bounding box
        # Returns
            :return: The transformed bounding box or None
        """

        # Transform the bbox
        tf_bbox = bbox.transform(transform=transform, min_transformed_bbox_size=4)

        if tf_bbox is None:
            return None

        # Check that the bbox contains the desired material sample color value - if not discard the bbox
        y_min = int(round(tf_bbox.y_min * pil_mask.height))
        x_min = int(round(tf_bbox.x_min * pil_mask.width))
        y_max = int(round(tf_bbox.y_max * pil_mask.height))
        x_max = int(round(tf_bbox.x_max * pil_mask.width))

        bbox_crop = pil_mask.crop(box=(x_min, y_min, x_max, y_max))

        if not image_utils.pil_image_band_contains_value(bbox_crop, band=0, val=material_sample.material_r_color):
            self.logger.debug_log('Material sample lost during transform: material_r_color: {}, original_bbox: {}, transformed_bbox: {}, found: {}'
                                  .format(material_sample.material_r_color, bbox.corners, tf_bbox.corners, image_utils.pil_image_get_unique_band_values(bbox_crop, 0)))
            return None

        return tf_bbox

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
        num_unlabeled_samples_in_batch = len(unlabeled_batch) if unlabeled_batch is not None else 0
        num_labeled_samples_in_batch = len(labeled_batch) if labeled_batch is not None else 0
        num_samples_in_batch = num_labeled_samples_in_batch + num_unlabeled_samples_in_batch

        #self.logger.debug_log('Batch crop shape: {}, resize shape: {}'.format(crop_shape, resize_shape))
        #self.logger.debug_log('Generating batch data for step {}: labeled: {}, ul: {}'.format(step_idx, labeled_batch, unlabeled_batch))

        stime = time.time()
        X, Y = self.get_labeled_batch_data(step_idx, labeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        X_unlabeled, Y_unlabeled = self.get_unlabeled_batch_data(step_idx, unlabeled_batch, crop_shape=crop_shape, resize_shape=resize_shape)
        self.logger.debug_log('Raw data generation took: {}s'.format(time.time() - stime))

        # Combine the data (labeled + unlabeled)
        X = X + X_unlabeled
        Y = Y + Y_unlabeled

        # Generate possible mean teacher data
        # Note: only applied to inputs ground truth must be the same
        X_teacher = None

        if self.generate_mean_teacher_data:
            stime_t_data = time.time()
            X_teacher = self._get_mean_teacher_data_from_image_batch(X, dtype=np.float32)
            self.logger.debug_log('Mean Teacher data generation took: {}s'.format(time.time()-stime_t_data))

        # Process all the information into numpy arrays
        X = np.array([img_to_array(img) for img in X], dtype=np.float32)
        Y = self._index_encode_batch_masks(Y, num_labeled=len(labeled_batch))
        W = self._create_batch_weights(Y, num_labeled=len(labeled_batch))

        # Normalize the photo batch data
        X = self._np_normalize_image_batch(X)

        # Apply possible gaussian noise
        # Note: applied after generating mean teacher data so teacher can have unnoised data during generation
        if self.use_data_augmentation:
            if self.data_augmentation_params.using_gaussian_noise:
                gaussian_noise_stddev = self.data_augmentation_params.gaussian_noise_stddev_function(step_idx)
                X += np.random.normal(loc=0.0, scale=gaussian_noise_stddev, size=X.shape)

        if self.batch_data_format == BatchDataFormat.SUPERVISED:
            batch_input_data = [X]
            batch_output_data = [np.expand_dims(Y, -1)]
        elif self.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # The dimensions of the number of unlabeled in the batch must match with batch dimension
            num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

            # Generate a dummy output for the dummy loss function and yield a batch of data
            dummy_output = np.zeros(shape=[num_samples_in_batch], dtype=np.int32)

            # Append the mean teacher data to the batch input data
            if self.generate_mean_teacher_data:
                if X_teacher is None:
                    raise ValueError('Supposed to generate teacher data but X_teacher is None cannot append data')
                batch_input_data = [X, Y, W, num_unlabeled, X_teacher]
            else:
                batch_input_data = [X, Y, W, num_unlabeled]

            # Provide the true classification masks for labeled samples only - these go to the second loss function
            # in the semi-supervised model that is only used to calculate metrics. The output has to have the same
            # rank as the output from the network. The output rank has to match the input rank so the masks need to
            # be expanded to the same rank.
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

        # If we are in debug mode, save the batch images
        if settings.DEBUG:
            self._debug_log_batch_data(step_idx=step_idx, X=X, Y=Y, X_teacher=X_teacher)

        return batch_input_data, batch_output_data

    def _index_encode_batch_masks(self, masks, num_labeled):
        # type: (list[PILImage], int) -> np.ndarray

        # We can assume all the masks in the batch to have the same spatial dimensions
        batch_size = len(masks)
        height = masks[0].height
        width = masks[0].width

        np_masks = np.zeros((batch_size, height, width), dtype=np.int32)

        for i in range(0, len(masks)):
            # Convert the mask to [HxW] numpy array
            np_mask = img_to_array(masks[i], dtype=np.int32)
            np_mask = np_mask.squeeze()

            # For the labeled
            if i < num_labeled:
                found_material_r_colors = np.unique(np_mask)

                for r_color in found_material_r_colors:
                    np_masks[i][np_mask == r_color] = self.material_r_color_to_material_class[r_color]
            # For the unlabeled
            else:
                np_masks[i] = np_mask

        return np_masks

    def _create_batch_weights(self, np_masks, num_labeled):
        # type: (np.ndarray, int) -> np.ndarray

        np_weights = np.ones_like(np_masks, dtype=np.float32)
        np_labeled_weights_view = np_weights[0:num_labeled]
        np_labeled_masks_view = np_masks[0:num_labeled]
        unique_class_ids = np.unique(np_labeled_masks_view)

        for class_id in unique_class_ids:
            np_labeled_weights_view[np_labeled_masks_view == class_id] = self.class_weights[class_id]

        return np_weights

    def _generate_mask_for_unlabeled_image(self, pil_img):
        # type: (PILImage) -> PILImage

        """
        Generates labels (mask) for unlabeled images. Either uses the default generator which is
        an all black image (all 0s) or the one specified during initialization. The label generator
        should encode the information in a binary mode with borders as 0s and everything else as >0
        in HxW format.

        # Arguments
            :param pil_img: the image as a PIL image
        # Returns
            :return: the generated mask as a PIL image (grayscale)
        """

        # If not using any segmentation function return default mask - all zeros same size as the image
        if self.superpixel_segmentation_function == SuperpixelSegmentationFunctionType.NONE:
            return PImage.new('L', pil_img.size, 0)

        # All superpixel masks are always saved in PNG to avoid any compression artefacts
        cached_mask_file_path = None

        # If this is an ImageFile it has a file name and it might be cached
        if self.superpixel_mask_cache_path is not None and isinstance(pil_img, PILImageFile):
            filename_no_ext = os.path.splitext(os.path.basename(pil_img.filename))[0]
            cached_mask_filename = '{}.png'.format(filename_no_ext)
            cached_mask_file_path = os.path.join(self.superpixel_mask_cache_path, cached_mask_filename)

            # Check if we can find the mask from the superpixel the mask cache
            if os.path.exists(cached_mask_file_path):
                try:
                    mask = image_utils.load_img(cached_mask_file_path, grayscale=True, num_read_attemps=2)
                    return mask
                except Exception as e:
                    self.logger.warn('Caught exception during superpixel caching (read): {}'.format(e.message))

        # Convert from the PIL image to numpy array
        np_img = image_utils.img_to_array(pil_img)

        # Equalize and ensure the image is in the desired format
        # Note: after equalization the values are in range [0, 1] so we
        # need to map them to [-1, 1] for segmentation
        np_img = image_utils.np_adaptive_histogram_equalization(np_img)
        np_img -= 0.5
        np_img *= 2.0

        # Generate the mask and cache it if we have a cache path
        if self.superpixel_segmentation_function == SuperpixelSegmentationFunctionType.FELZENSZWALB:
            np_mask = image_utils.np_get_felzenszwalb_segmentation(np_img, scale=1000, sigma=0.8, min_size=250, normalize_img=False, borders_only=True)
        elif self.superpixel_segmentation_function == SuperpixelSegmentationFunctionType.SLIC:
            np_mask = image_utils.np_get_slic_segmentation(np_img, n_segments=400, sigma=1, compactness=10.0, max_iter=20, normalize_img=False, borders_only=True)
        elif self.superpixel_segmentation_function == SuperpixelSegmentationFunctionType.QUICKSHIFT:
            np_mask = image_utils.np_get_quickshift_segmentation(np_img, kernel_size=10, max_dist=20, ratio=0.5, sigma=0.0, normalize_img=False, borders_only=True)
        elif self.superpixel_segmentation_function == SuperpixelSegmentationFunctionType.WATERSHED:
            np_mask = image_utils.np_get_watershed_segmentation(np_img, markers=400, compactness=0.0, normalize_img=False, borders_only=True)
        else:
            raise ValueError('Unknown label generation function type: {}'.format(self.superpixel_segmentation_function))

        # Expand the dimensions to make the mask into a grayscale image
        np_mask = np.expand_dims(np_mask, -1)
        np_mask = np_mask.astype(dtype=np.uint8)
        mask = image_utils.array_to_img(np_mask, scale=False)

        # If using caching save the mask
        if self.superpixel_mask_cache_path is not None and cached_mask_file_path is not None:
            try:
                mask.save(cached_mask_file_path, format='PNG')
            except Exception as e:
                self.logger.warn('Caught exception during superpixel caching (write): {}'.format(e.message))

        return mask

    def _debug_log_batch_data(self, step_idx, X, Y, X_teacher):
        if settings.DEBUG:
            b_min = np.min(X)
            b_max = np.max(X)
            b_min_teacher = np.min(X_teacher) if X_teacher is not None else 0.0
            b_max_teacher = np.max(X_teacher) if X_teacher is not None else 0.0

            for i in range(0, len(X)):
                img = ((X[i] - b_min) / (b_max - b_min)) * 255.0    # Scale photos to [0, 255]
                mask = Y[i][:, :, np.newaxis] * 10                  # Multiply mask values by 10 to make them visible
                self.logger.debug_log_image(img, '{}_{}_{}_photo.jpg'.format(self.name, step_idx, i), scale=False)
                self.logger.debug_log_image(mask, file_name='{}_{}_{}_mask.png'.format(self.name, step_idx, i), format='PNG', scale=False)

                if X_teacher is not None:
                    img = ((X_teacher[i] - b_min_teacher) / (b_max_teacher - b_min_teacher)) * 255.0
                    self.logger.debug_log_image(img, '{}_{}_{}_photo_teacher.jpg'.format(self.name, step_idx, i), scale=False)


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
                 params):
        # type: (MINCDataSet, UnlabeledImageDataSet, int, int, np.ndarray, DataGeneratorParameters) -> None

        self.labeled_data_set = labeled_data_set
        self.unlabeled_data_set = unlabeled_data_set
        self.num_labeled_per_batch = num_labeled_per_batch
        self.num_unlabeled_per_batch = num_unlabeled_per_batch
        self.class_weights = np.array(class_weights, dtype=np.float32)

        super(ClassificationDataGenerator, self).__init__(params)

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

        if self.using_unlabeled_data:
            X = X + X_unlabeled
            Y = Y + Y_unlabeled
            W = W + W_unlabeled

        self.logger.debug_log('Raw data generation took: {}s'.format(time.time() - stime))

        X_teacher = None

        # Generate possible mean teacher data
        # Note: only applied to inputs ground truth must be the same
        if self.generate_mean_teacher_data:
            stime_t_data = time.time()
            X_teacher = self._get_mean_teacher_data_from_image_batch(X, dtype=np.float32)
            self.logger.debug_log('Mean Teacher data generation took: {}s'.format(time.time()-stime_t_data))

        X = [img_to_array(img) for img in X]
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)

        num_unlabeled_samples_in_batch = len(X_unlabeled)
        num_samples_in_batch = len(X)

        # Normalize the photo batch data: color values to [-1, 1], subtract per pixel mean and divide by stddev
        X = self._np_normalize_image_batch(X)

        # Apply possible gaussian noise
        # Note: applied after generating mean teacher data so teacher can have unnoised data during generation
        if self.use_data_augmentation:
            if self.data_augmentation_params.using_gaussian_noise:
                gaussian_noise_stddev = self.data_augmentation_params.gaussian_noise_stddev_function(step_idx)
                X += np.random.normal(loc=0.0, scale=gaussian_noise_stddev, size=X.shape)

        if self.batch_data_format == BatchDataFormat.SUPERVISED:
            batch_input_data = [X]
            batch_output_data = [Y]
        elif self.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # The dimensions of the number of unlabeled in the batch must match with batch dimension
            num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

            # Generate a dummy output for the dummy loss function and yield a batch of data
            dummy_output = np.zeros(shape=[num_samples_in_batch], dtype=np.int32)

            # Append the mean teacher data to the batch input data
            if self.generate_mean_teacher_data:
                if X_teacher is None:
                    raise ValueError('Supposed to generate teacher data but X_teacher is None cannot append data')
                batch_input_data = [X, Y, W, num_unlabeled, X_teacher]
            else:
                batch_input_data = [X, Y, W, num_unlabeled]

            if X.shape[0] != Y.shape[0] or X.shape[0] != W.shape[0] or X.shape[0] != num_unlabeled.shape[0]:
                self.logger.warn('Unmatching input first (batch) dimensions: {}, {}, {}, {}'.format(X.shape[0], Y.shape[0], W.shape[0], num_unlabeled.shape[0]))

            logits_output = Y
            batch_output_data = [dummy_output, logits_output]
        else:
            raise ValueError('Unknown batch data format: {}'.format(self.batch_data_format))

        self.logger.debug_log('Data generation took in total: {}s'.format(time.time() - stime))

        # If we are in debug mode, save the batch images
        if settings.DEBUG:
            self._debug_log_batch_images(step_idx=step_idx, X=X, Y=Y, X_teacher=X_teacher, labeled_batch=labeled_batch, unlabeled_batch=unlabeled_batch)

        return batch_input_data, batch_output_data

    def get_labeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        if self.labeled_data_set.data_set_type == MINCDataSetType.MINC_2500:
            data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
                (delayed(pickle_method)(self,
                                        'get_labeled_sample_minc_2500',
                                        step_index=step_index,
                                        sample_index=sample_index,
                                        resize_shape=resize_shape) for sample_index in index_array)
        elif self.labeled_data_set.data_set_type == MINCDataSetType.MINC:
            data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
                (delayed(pickle_method)(self,
                                        'get_labeled_sample_minc',
                                        step_index=step_index,
                                        sample_index=sample_index,
                                        crop_shape=crop_shape,
                                        resize_shape=resize_shape) for sample_index in index_array)
        else:
            raise ValueError('Unknown data set type: {}'.format(self.labeled_data_set.data_set_type))

        X, Y, W = zip(*data)

        return X, Y, W

    def get_labeled_sample_minc_2500(self, step_index, sample_index, resize_shape):
        # type: (int, int, tuple[int, int]) -> (PILImage, np.ndarray, np.ndarray)

        minc_sample = self.labeled_data_set.samples[sample_index]
        img_file = self.labeled_data_set.photo_image_set.get_image_file_by_file_name(minc_sample.file_name)

        if img_file is None:
            raise ValueError('Could not find image from ImageSet with file name: {}'.format(minc_sample.file_name))

        img = img_file.get_image(self.num_color_channels)

        # Check whether we need to resize the photo to a constant size
        if resize_shape is not None:
            img = self._pil_resize_image(img, resize_shape=resize_shape, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC, img_type=ImageType.PHOTO)

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            images, _ = self._pil_apply_data_augmentation_to_images(images=[img],
                                                                    cvals=[self.photo_cval],
                                                                    random_seed=self.random_seed+step_index,
                                                                    interpolations=[ImageInterpolationType.BICUBIC])

            img, = images

        # Make sure the image dimensions satisfy the div2_constraint
        img = self._pil_fit_image_to_div2_constraint(img, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC)

        # Construct label vector (one-hot)
        custom_label = self.labeled_data_set.minc_label_to_custom_label[minc_sample.minc_label]
        y = np.zeros(self.labeled_data_set.num_classes, dtype=np.float32)
        y[custom_label] = 1.0

        # Construct weight vector
        w = self.class_weights

        return img, y, w

    def get_labeled_sample_minc(self, step_index, sample_index, crop_shape, resize_shape):
        # type: (int, int, tuple[int, int], tuple[int, int]) -> (PILImage, np.ndarray, np.ndarray)

        minc_sample = self.labeled_data_set.samples[sample_index]
        img_file = self.labeled_data_set.photo_image_set.get_image_file_by_file_name(minc_sample.file_name)

        if img_file is None:
            raise ValueError('Could not find image from ImageSet with file name: {}'.format(minc_sample.file_name))

        img = img_file.get_image(self.num_color_channels)

        if crop_shape is None:
            raise ValueError('MINC data set images cannot be used without setting a crop shape')

        # Check whether we need to resize the photo to a constant size
        if resize_shape is not None:
            img = self._pil_resize_image(img, resize_shape=resize_shape, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC, img_type=ImageType.PHOTO)

        img_height = img.height
        img_width = img.width
        crop_height = crop_shape[0]
        crop_width = crop_shape[1]
        crop_center_y = minc_sample.y
        crop_center_x = minc_sample.x

        if settings.DEBUG:
            img = image_utils.pil_draw_square(img, center_x=crop_center_x, center_y=crop_center_y, size=6, color=(255, 0, 0))

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            img_orig = img.copy()
            images, transform = self._pil_apply_data_augmentation_to_images(images=[img],
                                                                            cvals=[self.photo_cval],
                                                                            random_seed=self.random_seed+step_index,
                                                                            interpolations=[ImageInterpolationType.BICUBIC],
                                                                            transform_origin=np.array([crop_center_y, crop_center_x]))

            img, = images

            crop_center_new = transform.transform_normalized_coordinates(np.array([crop_center_x, crop_center_y]))
            crop_center_x_new, crop_center_y_new = crop_center_new[0], crop_center_new[1]

            # If the center has gone out of bounds abandon the augmentation - otherwise, update the crop center values
            if (not (0.0 < crop_center_y_new < 1.0)) or (not (0.0 < crop_center_x_new < 1.0)):
                # Destroy the augmented version and revert back to the copy of the original
                del img
                img = img_orig
            else:
                crop_center_y = crop_center_y_new
                crop_center_x = crop_center_x_new

                # Destroy the backup image
                del img_orig

        # Crop the image with the specified crop center. Regions going out of bounds are padded with a
        # constant value.
        y_c = crop_center_y*img_height
        x_c = crop_center_x*img_width

        # MINC crop values can go out of bounds so we can keep the crop size constant
        y_0 = int(round(y_c - (crop_height*0.5)))
        x_0 = int(round(x_c - (crop_width*0.5)))
        y_1 = int(round(y_c + (crop_height*0.5)))
        x_1 = int(round(x_c + (crop_width*0.5)))

        # Rounding can end up in off-by-one errors, fix them here
        # by moving the bottom right corner in x,y according to the difference
        crop_size_y = y_1 - y_0
        crop_diff_y = crop_height - crop_size_y
        y_1 += crop_diff_y

        crop_size_x = x_1 - x_0
        crop_diff_x = crop_width - crop_size_x
        x_1 += crop_diff_x

        # Final check for crop size
        crop_size_y = y_1 - y_0
        crop_size_x = x_1 - x_0

        if crop_size_y != crop_height or x_1 - x_0 != crop_width:
            raise ValueError('Mismatch in crop sizes in sample: {} with size: {}. Original center: {}, used center: {}, used crop coordinates: ({}, {}). Expected size: {}, got: {}.'.format(
                minc_sample.file_name,
                img.size,
                (minc_sample.x, minc_sample.y),
                (crop_center_x, crop_center_y),
                (x_0, y_0),
                (x_1, y_1),
                crop_shape,
                (crop_size_x, crop_size_y)))

        img = image_utils.pil_crop_image_with_fill(img, x1=x_0, y1=y_0, x2=x_1, y2=y_1, cval=self.photo_cval)
        img = self._pil_fit_image_to_div2_constraint(img=img, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC)

        # Construct label vector (one-hot)
        custom_label = self.labeled_data_set.minc_label_to_custom_label[minc_sample.minc_label]
        y = np.zeros(self.labeled_data_set.num_classes, dtype=np.float32)
        y[custom_label] = 1.0

        # Construct class weight vector
        w = self.class_weights

        return img, y, w

    def get_unlabeled_batch_data(self, step_index, index_array, crop_shape, resize_shape):
        if not self.using_unlabeled_data:
            return (), (), ()

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        data = Parallel(n_jobs=settings.DATA_GENERATION_THREADS_PER_PROCESS, backend='threading')\
            (delayed(pickle_method)
             (self,
              'get_unlabeled_sample',
              step_index=step_index,
              sample_index=sample_index,
              crop_shape=crop_shape,
              resize_shape=resize_shape) for sample_index in index_array)

        X, Y, W = zip(*data)

        return X, Y, W

    def get_unlabeled_sample(self, step_index, sample_index, crop_shape, resize_shape):
        # type: (int, int, tuple[int, int], tuple[int, int]) -> (PILImage, np.ndarray, np.ndarray)

        img_file = self.unlabeled_data_set.get_index(sample_index)
        img = img_file.get_image(self.num_color_channels)

        # Check whether we need to resize the photo and the mask to a constant size
        if resize_shape is not None:
            img = self._pil_resize_image(img, resize_shape=resize_shape, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC, img_type=ImageType.PHOTO)

        # Check whether any of the image dimensions is smaller than the crop,
        # if so pad with the assigned fill colors
        if crop_shape is not None and (img.height < crop_shape[0] or img.width < crop_shape[1]):
            # Image dimensions must be at minimum the same as the crop dimension
            # on each axis. The photo needs to be filled with the photo_cval
            min_img_shape = (max(crop_shape[0], img.height), max(crop_shape[1], img.width))
            img = image_utils.pil_pad_image_to_shape(img, min_img_shape, self.photo_cval)

        # Apply data augmentation
        if self._should_apply_augmentation(step_index):
            images, _ = self._pil_apply_data_augmentation_to_images(images=[img],
                                                                    cvals=[self.photo_cval],
                                                                    random_seed=self.random_seed+step_index,
                                                                    interpolations=[ImageInterpolationType.BICUBIC])
            img, = images

        # If a crop size is given: take a random crop of the image
        if crop_shape is not None:
            y1x1, y2x2 = self._get_random_crop_area(img_width=img.width, img_height=img.height, crop_width=crop_shape[1], crop_height=crop_shape[0])
            img = image_utils.pil_crop_image(img, x1=y1x1[1], y1=y1x1[0], x2=y2x2[1], y2=y2x2[0])

        img = self._pil_fit_image_to_div2_constraint(img=img, cval=self.photo_cval, interp=ImageInterpolationType.BICUBIC)

        # Create a dummy label vector (one-hot) all zeros
        y = self.dummy_label_vector

        # Construct class weight vector
        w = self.class_weights

        return img, y, w

    def _debug_log_batch_images(self, step_idx, X, Y, X_teacher, labeled_batch, unlabeled_batch):
        # type: (int, np.ndarray, np.ndarray, np.ndarray, list, list) -> None

        # If we are in debug mode, save the batch images
        if settings.DEBUG:
            b_min = np.min(X)
            b_max = np.max(X)
            b_min_teacher = np.min(X_teacher) if X_teacher is not None else 0
            b_max_teacher = np.max(X_teacher) if X_teacher is not None else 0

            names = [self.labeled_data_set.samples[x].photo_id for x in labeled_batch]
            names += [self.unlabeled_data_set.get_index(x).file_name for x in unlabeled_batch] if self.using_unlabeled_data else []

            for i in range(0, len(X)):
                label = np.argmax(Y[i])
                img = ((X[i] - b_min) / (b_max - b_min)) * 255.0
                self.logger.debug_log_image(img, '{}_{}_{}_{}_{}_photo.jpg'.format(label, self.name, step_idx, i, names[i]), scale=False)

                if X_teacher is not None:
                    img = ((X_teacher[i] - b_min_teacher) / (b_max_teacher - b_min_teacher)) * 255.0
                    self.logger.debug_log_image(img, '{}_{}_{}_{}_{}_photo_teacher.jpg'.format(label, self.name, step_idx, i, names[i]), scale=False)