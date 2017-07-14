# coding=utf-8

import random
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from PIL import Image

from utils import dataset_utils
from utils import image_utils
from utils.dataset_utils import MaterialClassInformation
from utils.thread_utils import threadsafe

from abc import ABCMeta, abstractmethod
import keras.backend as K
from joblib import Parallel, delayed
from typing import Callable


#######################################
# UTILITY FUNCTIONS
#######################################

def get_labeled_segmentation_data_pair(photo_mask_pair,
                                       num_color_channels,
                                       crop_shape,
                                       material_class_information,
                                       photo_cval,
                                       mask_cval,
                                       use_data_augmentation=False,
                                       data_augmentation_params=None,
                                       img_data_format='channels_last',
                                       div2_constraint=4,
                                       mask_type='one_hot'):
    # type: (tuple[str], int, tuple[int], list[MaterialClassInformation], np.array, np.array, bool, DataAugmentationParameters, str, int, str) -> (np.array, np.array)

    """
    Returns a photo mask pair for supervised segmentation training. Will apply data augmentation
    and cropping as instructed in the parameters.

    The photos are not normalized to range [-1,1] within the function.

    # Arguments
        :param photo_mask_pair: a pair of paths (path_to_photo, path_to_mask)
        :param num_color_channels: number of channels in the photos; 1, 3 or 4
        :param crop_shape: size of the crop or None if no cropping should be applied
        :param material_class_information: material class information to expand the mask
        :param photo_cval: fill color value for photos [0,255]
        :param mask_cval: fill color value for masks [0,255]
        :param use_data_augmentation: should data augmentation be used
        :param data_augmentation_params: parameters for data augmentation
        :param img_data_format: format of the image data 'channels_first' or 'channels_last'
        :param div2_constraint: divisibility constraint
    # Returns
        :return: a tuple of numpy arrays (image, mask)
    """

    # Load the image and mask as PIL images
    photo = dataset_utils.load_n_channel_image(photo_mask_pair[0], num_color_channels)
    mask = dataset_utils.load_n_channel_image(photo_mask_pair[1], 3)

    # Resize the photo to match the mask size if necessary, since
    # the original photos are sometimes huge
    if photo.size != mask.size:
        photo = photo.resize(mask.size, Image.ANTIALIAS)

    if photo.size != mask.size:
        raise ValueError('Non-matching photo and mask dimensions after resize: {} != {}'
                         .format(photo.size, mask.size))

    # Convert to numpy array
    np_photo = img_to_array(photo)
    np_mask = img_to_array(mask)

    # Apply crops and augmentation
    np_photo, np_mask = process_segmentation_photo_mask_pair(np_photo=np_photo,
                                                             np_mask=np_mask,
                                                             crop_shape=crop_shape,
                                                             photo_cval=photo_cval,
                                                             mask_cval=mask_cval,
                                                             use_data_augmentation=use_data_augmentation,
                                                             data_augmentation_params=data_augmentation_params,
                                                             img_data_format=img_data_format,
                                                             div2_constraint=div2_constraint)

    # Expand the mask image to the one-hot encoded shape: H x W x NUM_CLASSES
    if mask_type == 'one_hot':
        np_mask = dataset_utils.expand_mask(np_mask, material_class_information)
    elif mask_type == 'index':
        np_mask = dataset_utils.index_encode_mask(np_mask, material_class_information)
    else:
        raise ValueError('Unknown mask_type: {}'.format(mask_type))

    return np_photo, np_mask


def process_photo(np_photo,
                  crop_shape,
                  photo_cval,
                  use_data_augmentation=False,
                  data_augmentation_params=None,
                  img_data_format='channels_last',
                  div2_constraint=4):
    # type: (np.array, tuple[int], np.array, np.array, bool, str, int) -> np.array

    """
    Applies cropping and data augmentation to a single photo.

    # Arguments
        :param np_photo: the photo as a numpy array
        :param crop_shape: size of the crop or None if no cropping should be applied
        :param photo_cval: fill color value for photos [0,255]
        :param use_data_augmentation: should data augmentation be used
        :param data_augmentation_params: parameters for data augmentation
        :param img_data_format: format of the image data 'channels_first' or 'channels_last'
        :param div2_constraint: divisibility constraint
    # Returns
        :return: the augmented photo as a numpy array
    """

    # Check whether any of the image dimensions is smaller than the crop,
    # if so pad with the assigned fill colors
    if crop_shape is not None and (np_photo.shape[1] < crop_shape[0] or np_photo.shape[0] < crop_shape[1]):
        # Image dimensions must be at minimum the same as the crop dimension
        # on each axis. The photo needs to be filled with the photo_cval color and masks
        # with the mask cval color
        min_img_shape = (max(crop_shape[1], np_photo.shape[0]), max(crop_shape[0], np_photo.shape[1]))
        np_photo = image_utils.np_pad_image_to_shape(np_photo, min_img_shape, photo_cval)

    # If we are using data augmentation apply the random transformation
    if use_data_augmentation and np.random.random() <= data_augmentation_params.augmentation_probability:
        np_photo, = image_utils.np_apply_random_transform(images=[np_photo],
                                                          cvals=[photo_cval],
                                                          fill_mode=data_augmentation_params.fill_mode,
                                                          img_data_format=img_data_format,
                                                          rotation_range=data_augmentation_params.rotation_range,
                                                          zoom_range=data_augmentation_params.zoom_range,
                                                          horizontal_flip=data_augmentation_params.horizontal_flip,
                                                          vertical_flip=data_augmentation_params.vertical_flip)

    # If a crop size is given: take a random crop of both the image and the mask
    if crop_shape is not None:
        if dataset_utils.count_trailing_zeroes(crop_shape[0]) < div2_constraint or \
                        dataset_utils.count_trailing_zeroes(crop_shape[1]) < div2_constraint:
            raise ValueError('The crop size does not satisfy the div2 constraint of {}'.format(div2_constraint))

        x1y1, x2y2 = image_utils.np_get_random_crop_area(np_photo, crop_shape[0], crop_shape[1])
        np_photo = image_utils.np_crop_image(np_photo, x1y1[0], x1y1[1], x2y2[0], x2y2[1])
    else:
        # If a crop size is not given, make sure the image dimensions satisfy
        # the div2_constraint i.e. are n times divisible by 2 to work within
        # the network. If the dimensions are not ok pad the images.
        img_height_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[0])
        img_width_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[1])

        if img_height_div2 < div2_constraint or img_width_div2 < div2_constraint:

            # The photo needs to be filled with the photo_cval color
            padded_shape = dataset_utils.get_required_image_dimensions(np_photo.shape, div2_constraint)
            np_photo = image_utils.np_pad_image_to_shape(np_photo, padded_shape, photo_cval)

    return np_photo


def process_segmentation_photo_mask_pair(np_photo,
                                         np_mask,
                                         crop_shape,
                                         photo_cval,
                                         mask_cval,
                                         use_data_augmentation=False,
                                         data_augmentation_params=None,
                                         img_data_format='channels_last',
                                         div2_constraint=4):
    # type: (np.array, np.array, tuple[int], np.array, np.array, bool, DataAugmentationParameters, str, int) -> (np.array, np.array)

    """
    Applies crop and data augmentation to two numpy arrays representing the photo and
    the respective segmentation mask. The photos are not normalized to range [-1,1]
    within the function.

    # Arguments
        :param np_photo: the photo as a numpy array
        :param np_mask: the mask as a numpy array must have same spatial dimensions (HxW) as np_photo
        :param crop_shape: size of the crop or None if no cropping should be applied
        :param photo_cval: fill color value for photos [0,255]
        :param mask_cval: fill color value for masks [0,255]
        :param use_data_augmentation: should data augmentation be used
        :param data_augmentation_params: parameters for data augmentation
        :param img_data_format: format of the image data 'channels_first' or 'channels_last'
        :param div2_constraint: divisibility constraint
    # Returns
        :return: a tuple of numpy arrays (image, mask)
    """

    if np_photo.shape[:2] != np_mask.shape[:2]:
        raise ValueError('Non-matching photo and mask shapes: {} != {}'.format(np_photo.shape, np_mask.shape))

    # Check whether any of the image dimensions is smaller than the crop,
    # if so pad with the assigned fill colors
    if crop_shape is not None and (np_photo.shape[1] < crop_shape[0] or np_photo.shape[0] < crop_shape[1]):
        # Image dimensions must be at minimum the same as the crop dimension
        # on each axis. The photo needs to be filled with the photo_cval color and masks
        # with the mask cval color
        min_img_shape = (max(crop_shape[1], np_photo.shape[0]), max(crop_shape[0], np_photo.shape[1]))
        np_photo = image_utils.np_pad_image_to_shape(np_photo, min_img_shape, photo_cval)
        np_mask = image_utils.np_pad_image_to_shape(np_mask, min_img_shape, mask_cval)

    # If we are using data augmentation apply the random transformation
    # to both the image and mask now. We apply the transformation to the
    # whole image to decrease the number of 'dead' pixels due to transformations
    # within the possible crop.
    if use_data_augmentation and np.random.random() <= data_augmentation_params.augmentation_probability:
        np_photo, np_mask = image_utils.np_apply_random_transform(images=[np_photo, np_mask],
                                                                  cvals=[photo_cval, mask_cval],
                                                                  fill_mode=data_augmentation_params.fill_mode,
                                                                  img_data_format=img_data_format,
                                                                  rotation_range=data_augmentation_params.rotation_range,
                                                                  zoom_range=data_augmentation_params.zoom_range,
                                                                  horizontal_flip=data_augmentation_params.horizontal_flip,
                                                                  vertical_flip=data_augmentation_params.vertical_flip)

    # If a crop size is given: take a random crop of both the image and the mask
    if crop_shape is not None:
        if dataset_utils.count_trailing_zeroes(crop_shape[0]) < div2_constraint or \
                        dataset_utils.count_trailing_zeroes(crop_shape[1]) < div2_constraint:
            raise ValueError('The crop size does not satisfy the div2 constraint of {}'.format(div2_constraint))

        # Re-attempt crops if the crops end up getting only background
        # i.e. black pixels
        attempts = 5

        while attempts > 0:
            x1y1, x2y2 = image_utils.np_get_random_crop_area(np_mask, crop_shape[0], crop_shape[1])
            mask_crop = image_utils.np_crop_image(np_mask, x1y1[0], x1y1[1], x2y2[0], x2y2[1])

            # If the mask crop is only background (all R channel is zero) - try another crop
            # until attempts are exhausted
            if np.max(mask_crop[:, :, 0]) == 0 and attempts-1 != 0:
                attempts -= 1
                continue

            np_mask = mask_crop
            np_photo = image_utils.np_crop_image(np_photo, x1y1[0], x1y1[1], x2y2[0], x2y2[1])
            break

    # If a crop size is not given, make sure the image dimensions satisfy
    # the div2_constraint i.e. are n times divisible by 2 to work within
    # the network. If the dimensions are not ok pad the images.
    img_height_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[0])
    img_width_div2 = dataset_utils.count_trailing_zeroes(np_photo.shape[1])

    if img_height_div2 < div2_constraint or img_width_div2 < div2_constraint:

        # The photo needs to be filled with the photo_cval color and masks with the mask cval color
        padded_shape = dataset_utils.get_required_image_dimensions(np_photo.shape, div2_constraint)
        np_photo = image_utils.np_pad_image_to_shape(np_photo, padded_shape, photo_cval)
        np_mask = image_utils.np_pad_image_to_shape(np_mask, padded_shape, mask_cval)

    return np_photo, np_mask


# Workaround to make the calls to label generation pickable
def _generate_labels_for_unlabeled_photo(np_photo, func):
    # type: (np.array[float32], Callable[[np.array[float32]], np.array]) -> np.array
    return func(np_photo)


#######################################
# UTILITY CLASSES
#######################################

class DataAugmentationParameters:
    """
    This class helps to maintain the data augmentation parameters for data generators.
    """

    def __init__(self,
                 augmentation_probability=0.5,
                 rotation_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 fill_mode='constant'):
        """
        # Arguments
            :param augmentation_probability: probability with which to apply random augmentations
            :param rotation_range: range of random rotations
            :param zoom_range: range of random zooms
            :param horizontal_flip: should horizontal flips be applied
            :param vertical_flip: should vertical flips be applied
            :param fill_mode: how should we fill overgrown areas
        # Returns
            :return: A new instance of DataAugmentationParameters
        """

        self.augmentation_probability = augmentation_probability
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode

        # Zoom range can either be a tuple or a scalar
        if np.isscalar(zoom_range):
            self.zoom_range = np.array([1.0 - zoom_range, 1.0 + zoom_range])
        elif len(zoom_range) == 2:
            self.zoom_range = np.array([zoom_range[0], zoom_range[1]])
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)


class DataGeneratorParameters(object):
    """
    This class helps to maintain parameters in common with all different data generators.
    """

    def __init__(self,
                 material_class_information,
                 num_color_channels,
                 random_seed=None,
                 use_per_channel_mean_normalization=True,
                 per_channel_mean=None,
                 use_per_channel_stddev_normalization=True,
                 per_channel_stddev=None,
                 photo_cval=None,
                 mask_cval=None,
                 use_data_augmentation=False,
                 data_augmentation_params=None,
                 shuffle_data_after_epoch=True):

        self.material_class_information = material_class_information
        self.num_color_channels = num_color_channels
        self.random_seed = random_seed
        self.use_per_channel_mean_normalization = use_per_channel_mean_normalization
        self.per_channel_mean = per_channel_mean
        self.use_per_channel_stddev_normalization = use_per_channel_stddev_normalization
        self.per_channel_stddev = per_channel_stddev
        self.photo_cval = photo_cval
        self.mask_cval = mask_cval
        self.use_data_augmentation = use_data_augmentation
        self.data_augmentation_params = data_augmentation_params
        self.shuffle_data_after_epoch = shuffle_data_after_epoch


#######################################
# DATA GENERATOR
#######################################

class DataGenerator(object):

    """
    Abstract class which declares necessary methods for different DataGenerators. Also,
    unwraps the DataGeneratorParameters to class member variables.
    """

    __metaclass__ = ABCMeta

    def __init__(self, params):
        # type: (DataGeneratorParameters) -> ()

        # Unwrap DataGeneratorParameters to member variables
        self.material_class_information = params.material_class_information
        self.num_color_channels = params.num_color_channels
        self.random_seed = params.random_seed
        self.use_per_channel_mean_normalization = params.use_per_channel_mean_normalization
        self.per_channel_mean = params.per_channel_mean
        self.use_per_channel_stddev_normalization = params.use_per_channel_stddev_normalization
        self.per_channel_stddev = params.per_channel_stddev
        self.photo_cval = params.photo_cval
        self.mask_cval =params.mask_cval
        self.use_data_augmentation = params.use_data_augmentation
        self.data_augmentation_params = params.data_augmentation_params
        self.shuffle_data_after_epoch = params.shuffle_data_after_epoch

        # Other member variables
        self.img_data_format = K.image_data_format()

        # Ensure the per_channel_mean is a numpy tensor
        if self.per_channel_mean is not None:
            self.per_channel_mean = np.array(self.per_channel_mean)

        # Ensure per_channel_stddev is a numpy tensor
        if self.per_channel_stddev is not None:
            self.per_channel_stddev = np.array(self.per_channel_stddev)

        # Calculate missing per-channel mean if necessary
        if self.use_per_channel_mean_normalization and self.per_channel_mean is None:
            photos = self.get_all_photos()
            self.per_channel_mean = dataset_utils.calculate_per_channel_mean(photos, self.num_color_channels)
            print 'DataGenerator: Using per-channel mean: {}'.format(self.per_channel_mean)

        # Calculate missing per-channel stddev if necessary
        if self.use_per_channel_stddev_normalization and self.per_channel_stddev is None:
            photos = self.get_all_photos()
            self.per_channel_stddev = dataset_utils.calculate_per_channel_stddev(photos, self.per_channel_mean, self.num_color_channels)
            print 'DataGenerator: Using per-channel stddev: {}'.format(self.per_channel_stddev)

        # Use per-channel mean but in range [0, 255] if nothing else is given.
        # The normalization is done to the whole batch after transformations so
        # the images are not in range [-1,1] before transformations.
        if self.photo_cval is None:
            if self.per_channel_mean is None:
                self.photo_cval = np.array([0.0] * 3)
            else:
                self.photo_cval = dataset_utils.np_from_normalized_to_255(np.array(self.per_channel_mean))
            print 'DataGenerator: Using photo cval: {}'.format(self.photo_cval)

        # Use black (background)
        if self.mask_cval is None:
            self.mask_cval = np.array([0.0] * 3)
            print 'DataGenerator: Using mask cval: {}'.format(self.mask_cval)

        # Use the given random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    @abstractmethod
    def get_all_photos(self):
        # type: () -> list[str]

        """
        Returns all the photo file paths as a list

        # Arguments
            None
        # Returns
            :return: all the photo file paths as a list
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    @abstractmethod
    def get_all_masks(self):
        # type: () -> list[str]

        """
        Returns all the mask file paths as a list.

        # Arguments
            None
        # Returns
            :return: all the mask file paths as a list
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')

    @abstractmethod
    def get_flow(self):
        # type: () -> ()

        """
        This method should yield batches of data. The arguments for derived classes might vary.

        # Arguments
            None
        # Returns
            None
        """
        raise NotImplementedError('This is not implemented within the abstract DataGenerator class')


#######################################
# SEGMENTATION DATA GENERATOR
#######################################

class SegmentationDataGenerator(DataGenerator):
    """
    DataGenerator which provides supervised segmentation data. Produces batches
    of matching image segmentation mask pairs.
    """

    def __init__(
            self,
            labeled_data,
            params):
        # type: (list[tuple[str]], DataGeneratorParameters) -> ()

        self.labeled_data = labeled_data
        super(SegmentationDataGenerator, self).__init__(params)

    def get_all_photos(self):
        # type: () -> list[str]

        """
        Returns all the photo file paths as a list

        # Arguments
            None
        # Returns
            :return: all the photo file paths as a list
        """
        return [photo_mask_pair[0] for photo_mask_pair in self.labeled_data]

    def get_all_masks(self):
        # type: () -> list[str]

        """
        Returns all the mask file paths as a list.

        # Arguments
            None
        # Returns
            :return: all the mask file paths as a list
        """
        return [photo_mask_pair[1] for photo_mask_pair in self.labeled_data]

    @threadsafe
    def get_flow(self, batch_size, crop_shape=None):
        # type: (int, tuple[int]) -> (np.array, np.array)

        """
        Yields batches of data endlessly.

        # Arguments
            :param batch_size: size of a single minibatch
            :param crop_shape: shape of the crop WxH, None if no cropping should be applied
        # Returns
            :return: batch of input image, segmentation mask data as a tuple (X,Y)
        """

        # Calculate the number of batches. If the length of the data is not
        # divisible by the batch size then the last batch will be smaller
        # than the rest. However, all data is used on every epoch.
        data_set_size = len(self.labeled_data)
        num_batches = dataset_utils.get_number_of_batches(data_set_size, batch_size)
        n_jobs = dataset_utils.get_number_of_parallel_jobs()

        while True:
            if self.shuffle_data_after_epoch:
                # Shuffle the (photo, mask) -pairs after every epoch
                random.shuffle(self.labeled_data)

            for i in range(0, num_batches):
                # The files for this batch
                batch_range = dataset_utils.get_batch_index_range(data_set_size, batch_size, i)
                batch_files = self.labeled_data[batch_range[0]:batch_range[1]]

                # Parallel processing of the files in this batch
                data = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(get_labeled_segmentation_data_pair)(
                        photo_mask_pair=photo_mask_pair,
                        num_color_channels=self.num_color_channels,
                        crop_shape=crop_shape,
                        material_class_information=self.material_class_information,
                        photo_cval=self.photo_cval,
                        mask_cval=self.mask_cval,
                        use_data_augmentation=self.use_data_augmentation,
                        data_augmentation_params=self.data_augmentation_params,
                        img_data_format=self.img_data_format,
                        div2_constraint=4,
                        mask_type='one_hot') for photo_mask_pair in batch_files)

                # Note: all the examples in the batch have to have the same dimensions
                X, Y = zip(*data)
                X, Y = np.array(X), np.array(Y)

                # Normalize the photo batch data
                X = dataset_utils \
                    .normalize_batch(X,
                                     self.per_channel_mean if self.use_per_channel_mean_normalization else None,
                                     self.per_channel_stddev if self.use_per_channel_stddev_normalization else None)

                yield X, Y


################################################
# SEMI SUPERVISED SEGMENTATION DATA GENERATOR
################################################


class SemisupervisedSegmentationDataGenerator(DataGenerator):

    def __init__(self,
                 labeled_data,
                 unlabeled_data,
                 params,
                 label_generation_function=None):
        # type: (list[tuple[str]], list[str], DataGeneratorParameters, Callable[[np.array[np.float32]], np.array]) -> ()

        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

        if label_generation_function is None:
            self.label_generation_function = SemisupervisedSegmentationDataGenerator.default_label_generator_for_unlabeled_photos
        else:
            self.label_generation_function = label_generation_function

        super(SemisupervisedSegmentationDataGenerator, self).__init__(params)

    def get_all_photos(self):
        # type: () -> list[str]

        """
        Returns all the photo file paths as a list

        # Arguments
            None
        # Returns
            :return: all the photo file paths as a list
        """
        photos = [photo_mask_pair[0] for photo_mask_pair in self.labeled_data]
        if self.unlabeled_data is not None:
            photos += self.unlabeled_data
        return photos

    def get_all_masks(self):
        # type: () -> list[str]

        """
        Returns all the mask file paths as a list.

        # Arguments
            None
        # Returns
            :return: all the mask file paths as a list
        """
        masks = [photo_mask_pair[1] for photo_mask_pair in self.labeled_data]
        return masks

    @threadsafe
    def get_flow(self, num_labeled_per_batch, num_unlabeled_per_batch, crop_shape=None):
        # type: (int, int, tuple(int)) -> list[np.array, np.array, np.array, np.array, np.array]

        """
        Generates batches of data for semi supervised mean teacher training. Also handles
        updating the mean teacher model after every batch of data.

        # Arguments
            :param num_labeled_per_batch: number of labeled images per batch
            :param num_unlabeled_per_batch: number of unlabeled images per batch
            :param crop_shape: crop size of the images (WxH)
        # Returns
            :return: A list of inputs in the following order

            0: Input images consisting of both labeled and unlabeled images. The unlabeled images are
               the M last images in the batch.
            1: Labels of the input images. This set has labels for both labeled and unlabeled images
               and the unlabeled images.
            2: Number of unlabeled data in the X and Y (photos and labels).
        """
        # Calculate the number of batches. If the length of the labeled data is not
        # divisible by the labeled data batch size then the last batch will be smaller
        # than the rest. However, all labeled data is used on every epoch.
        labeled_data_set_size = len(self.labeled_data)

        if self.unlabeled_data is not None:
            unlabeled_data_set_size = len(self.unlabeled_data)
        else:
            unlabeled_data_set_size = 0

        total_batch_size = num_labeled_per_batch + num_unlabeled_per_batch
        num_batches = dataset_utils.get_number_of_batches(labeled_data_set_size, num_labeled_per_batch)
        n_jobs = dataset_utils.get_number_of_parallel_jobs()

        while True:
            if self.shuffle_data_after_epoch:
                # Shuffle the labeled (photo, mask) data after every epoch
                random.shuffle(self.labeled_data)

            for i in range(0, num_batches):
                labeled_batch_range = dataset_utils.get_batch_index_range(labeled_data_set_size, num_labeled_per_batch, i)
                labeled_batch_files = self.labeled_data[labeled_batch_range[0]:labeled_batch_range[1]]
                num_labeled_in_batch = len(labeled_batch_files)

                if self.unlabeled_data is not None and num_unlabeled_per_batch > 0:
                    unlabeled_batch_range = dataset_utils.get_batch_index_range(unlabeled_data_set_size, num_unlabeled_per_batch, i)
                    unlabeled_batch_files = self.unlabeled_data[unlabeled_batch_range[0]:unlabeled_batch_range[1]]
                    num_unlabeled_in_batch = len(unlabeled_batch_files)
                else:
                    num_unlabeled_in_batch = 0

                samples_in_batch = num_labeled_in_batch + num_unlabeled_in_batch

                # In parallel: Process the labeled files for this batch
                labeled_data = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(get_labeled_segmentation_data_pair)(
                        photo_mask_pair=photo_mask_pair,
                        num_color_channels=self.num_color_channels,
                        crop_shape=crop_shape,
                        material_class_information=self.material_class_information,
                        photo_cval=self.photo_cval,
                        mask_cval=self.mask_cval,
                        use_data_augmentation=self.use_data_augmentation,
                        data_augmentation_params=self.data_augmentation_params,
                        img_data_format=self.img_data_format,
                        div2_constraint=4,
                        mask_type='index') for photo_mask_pair in labeled_batch_files)

                if num_labeled_in_batch > 0:
                    # In parallel: Load the unlabeled photos of this batch into numpy arrays
                    np_unlabeled_photos = Parallel(n_jobs=n_jobs, backend='threading')(
                        delayed(img_to_array)(dataset_utils.load_n_channel_image(
                            unlabeled_photo_path, self.num_color_channels)) for unlabeled_photo_path in unlabeled_batch_files)

                    # In parallel: Process the unlabeled data pairs (take crops, apply data augmentation, etc).
                    X_unlabeled = Parallel(n_jobs=n_jobs, backend='threading')(
                        delayed(process_photo)(
                            np_photo=np_unlabeled_photos[i],
                            crop_shape=crop_shape,
                            photo_cval=self.photo_cval,
                            use_data_augmentation=self.use_data_augmentation,
                            data_augmentation_params=self.data_augmentation_params,
                            img_data_format=self.img_data_format,
                            div2_constraint=4) for i in range(0, num_unlabeled_in_batch))

                    # Check whether we ran out of unlabeled data and shuffle if required
                    if unlabeled_batch_range[1] == unlabeled_data_set_size or num_unlabeled_in_batch < num_unlabeled_per_batch:
                        if self.shuffle_data_after_epoch:
                            random.shuffle(self.unlabeled_data)

                # Combine the labeled and unlabeled together so that the unlabeled samples are the M last
                # samples in the batch
                X, Y = zip(*labeled_data)

                if num_unlabeled_per_batch > 0:
                    # In parallel: Generate segmentation masks for the unlabeled photos
                    # Note: cropping and augmentation already applied, but channels not normalized
                    Y_unlabeled = Parallel(n_jobs=n_jobs, backend='threading')(
                        delayed(_generate_labels_for_unlabeled_photo)(
                            np_photo, self.label_generation_function) for np_photo in X_unlabeled)

                    X = list(X) + list(X_unlabeled)
                    Y = list(Y) + list(Y_unlabeled)

                X, Y = np.array(X), np.array(Y)

                # Normalize the photo batch data
                X = dataset_utils\
                    .normalize_batch(X,
                                     self.per_channel_mean if self.use_per_channel_mean_normalization else None,
                                     self.per_channel_stddev if self.use_per_channel_stddev_normalization else None)

                # The dimensions of the number of unlabeled in the batch must match with batch dimension
                num_unlabeled = np.ones(shape=[samples_in_batch], dtype=np.int32) * num_unlabeled_in_batch

                # Generate a dummy output for the dummy loss function and yield a batch of data
                dummy_output = np.zeros(shape=[samples_in_batch])

                batch_data = [X, Y, num_unlabeled]
                yield batch_data, dummy_output

    @staticmethod
    def default_label_generator_for_unlabeled_photos(np_image):
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

# Dirty trick to make pickling possible for member functions, see:
# https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma
def unwrap_get_data_pair(arg, **kwarg):
    return ClassificationDataGenerator.get_data_pair(*arg, **kwarg)


class ClassificationDataGenerator(object):
    """
    DataGenerator which provides supervised classification data. Produces batches
    of matching image one-hot-encoded vector pairs.
    """

    def __init__(self,
                 path_to_images_folder,
                 per_channel_mean,
                 path_to_labels_file,
                 path_to_categories_file,
                 num_channels,
                 random_seed=None,
                 verbose=False,
                 shuffle_data_after_epoch=True):

        self.path_to_images_folder = path_to_images_folder
        self.per_channel_mean = per_channel_mean
        self.path_to_labels_file = path_to_labels_file
        self.path_to_categories_file = path_to_categories_file
        self.num_channels = num_channels
        self.verbose = verbose
        self.shuffle_data_after_epoch = shuffle_data_after_epoch

        # Read categories file and build the categories dictionary
        # category name -> category index
        self.categories = {}

        with open(path_to_categories_file, 'r') as f:
            lines = f.readlines()
            # Remove whitespace characters like `\n` at the end of each line
            lines = [x.strip() for x in lines]

            for idx, category in enumerate(lines):
                self.categories[category] = idx

        self.num_categories = len(self.categories)

        if self.verbose:
            print 'Found {} categories: {}'.format(self.num_categories, self.categories)

        # Read the image file paths into an array
        with open(path_to_labels_file, 'r') as f:
            self.files = f.readlines()
            # Remove whitespace characters like `\n` at the end of each line
            self.files = [x.strip() for x in self.files]

        self.num_samples = len(self.files)

        if self.verbose:
            print 'Found {} samples'.format(self.num_samples)

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed()

    def get_data_pair(self, f):
        # type: (str) -> (np.array, np.array)

        """
        Creates a data pair of image and one-hot-encoded category vector from a MINC
        photo path.

        # Arguments
            :param f: path to MINC photo
        # Returns
            :return: a tuple of numpy arrays (image, one-hot-encoded category vector)
        """

        # The path is always /images/<category>/<filename>
        file_parts = f.split('/')
        category = file_parts[1]
        filename = file_parts[2]

        # Build the complete path
        file_path = os.path.join(self.path_to_images_folder, category, filename)

        # Load the image
        img = dataset_utils.load_n_channel_image(file_path, self.num_channels)
        x = img_to_array(img)

        # Create the one-hot-vector for the classification
        category_index = self.categories[category]
        y = np.zeros(self.num_categories)
        y[category_index] = 1.0

        return x, y

    @threadsafe
    def get_flow(self, batch_size):

        dataset_size = len(self.files)
        num_batches = dataset_utils.get_number_of_batches(dataset_size, batch_size)
        n_jobs = dataset_utils.get_number_of_parallel_jobs()

        while True:
            if self.shuffle_data_after_epoch:
                # Shuffle the files
                random.shuffle(self.files)

            for i in range(0, num_batches):
                # Get the files for this batch
                batch_range = dataset_utils.get_batch_index_range(dataset_size, batch_size, i)
                files = self.files[batch_range[0]:batch_range[1]]

                data = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(unwrap_get_data_pair)((self, f)) for f in files)

                X, Y = zip(*data)
                X, Y = dataset_utils.normalize_batch(np.array(X)), np.array(Y)

                yield X, Y
