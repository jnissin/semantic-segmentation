# coding=utf-8

import random
import numpy as np
import threading
import warnings
from enum import Enum

from keras.preprocessing.image import img_to_array
from PIL import Image

from utils import dataset_utils
from utils import image_utils
from utils.dataset_utils import MaterialClassInformation, MaterialSample
from data_set import LabeledImageDataSet, UnlabeledImageDataSet, ImageFile

from abc import ABCMeta, abstractmethod, abstractproperty
import keras.backend as K


#######################################
# UTILITY FUNCTIONS
#######################################

def get_labeled_segmentation_data_pair(photo_mask_pair,
                                       material_sample,
                                       num_color_channels,
                                       crop_shape,
                                       resize_shape,
                                       material_class_information,
                                       photo_cval,
                                       mask_cval,
                                       use_data_augmentation=False,
                                       data_augmentation_params=None,
                                       img_data_format='channels_last',
                                       div2_constraint=4,
                                       mask_type='one_hot'):
    # type: (tuple[ImageFile], MaterialSample, int, tuple[int], tuple[int], list[MaterialClassInformation], np.array, np.array, bool, DataAugmentationParameters, str, int, str) -> (np.array, np.array)

    """
    Returns a photo mask pair for supervised segmentation training. Will apply data augmentation
    and cropping as instructed in the parameters.

    The photos are not normalized to range [-1,1] within the function.

    # Arguments
        :param photo_mask_pair: a pair of ImageFiles (photo, mask)
        :param material_sample: material sample information for the files
        :param num_color_channels: number of channels in the photos; 1, 3 or 4
        :param crop_shape: size of the crop or None if no cropping should be applied
        :param resize_shape: size of the desired rezied images, None if no resizing should be applied
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
    photo = photo_mask_pair[0].get_image(color_channels=num_color_channels)
    mask = photo_mask_pair[1].get_image(color_channels=3)

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
                                                             material_sample=material_sample,
                                                             crop_shape=crop_shape,
                                                             resize_shape=resize_shape,
                                                             photo_cval=photo_cval,
                                                             mask_cval=mask_cval,
                                                             use_data_augmentation=use_data_augmentation,
                                                             data_augmentation_params=data_augmentation_params,
                                                             img_data_format=img_data_format,
                                                             div2_constraint=div2_constraint,
                                                             retry_crops=True)

    # Expand the mask image to the one-hot encoded shape: H x W x NUM_CLASSES
    if mask_type == 'one_hot':
        np_mask = dataset_utils.expand_mask(np_mask, material_class_information)
    elif mask_type == 'index':
        np_mask = dataset_utils.index_encode_mask(np_mask, material_class_information)
    else:
        raise ValueError('Unknown mask_type: {}'.format(mask_type))

    return np_photo, np_mask


def get_unlabeled_segmentation_data_pair(photo,
                                         label_generation_function,
                                         num_color_channels,
                                         crop_shape,
                                         resize_shape,
                                         photo_cval,
                                         mask_cval,
                                         use_data_augmentation=False,
                                         data_augmentation_params=None,
                                         img_data_format='channels_last',
                                         div2_constraint=4):
    # type: (ImageFile, function, int, tuple[int], tuple[int], np.array, np.array, bool, DataAugmentationParameters, str, int, str) -> (np.array, np.array)

    """
    Returns a photo mask pair for semi-supervised/unsupervised segmentation training.
    Will apply data augmentation and cropping as instructed in the parameters.

    The photos are not normalized to range [-1,1] within the function.

    # Arguments
        :param photo: an ImageFile of the photo
        :param label_generation_function: a function that takes a numpy array as a parameter and generates labels
        :param num_color_channels: number of channels in the photos; 1, 3 or 4
        :param crop_shape: size of the crop or None if no cropping should be applied
        :param resize_shape: size of the desired rezied images, None if no resizing should be applied
        :param photo_cval: fill color value for photos [0,255]
        :param mask_cval: fill color value for masks [0,255]
        :param use_data_augmentation: should data augmentation be used
        :param data_augmentation_params: parameters for data augmentation
        :param img_data_format: format of the image data 'channels_first' or 'channels_last'
        :param div2_constraint: divisibility constraint
    # Returns
        :return: a tuple of numpy arrays (image, mask)
    """

    # Load the photo as PIL image
    photo = photo.get_image(color_channels=num_color_channels)
    np_photo = img_to_array(photo)

    # Generate mask for the photo - note: the labels are generated before cropping
    # and augmentation to capture global structure within the image
    np_mask = label_generation_function(np_photo)

    # Expand the last dimension of the mask to make it compatible with augmentation functions
    np_mask = np_mask[:, :, np.newaxis]

    # Apply crops and augmentation
    np_photo, np_mask = process_segmentation_photo_mask_pair(np_photo=np_photo,
                                                             np_mask=np_mask,
                                                             material_sample=None,
                                                             crop_shape=crop_shape,
                                                             resize_shape=resize_shape,
                                                             photo_cval=photo_cval,
                                                             mask_cval=mask_cval,
                                                             use_data_augmentation=use_data_augmentation,
                                                             data_augmentation_params=data_augmentation_params,
                                                             img_data_format=img_data_format,
                                                             div2_constraint=div2_constraint,
                                                             retry_crops=False)

    # Squeeze the unnecessary last dimension out
    np_mask = np.squeeze(np_mask)

    # Map the mask values back to a continuous range [0, N_SUPERPIXELS]. The values
    # might be non-continuous due to cropping and augmentation
    old_indices = np.unique(np_mask)
    new_indices = np.arange(np.max(old_indices+1))

    for i in range(0, len(old_indices)):
        index_mask = np_mask[:, :] == old_indices[i]
        np_mask[index_mask] = new_indices[i]

    return np_photo, np_mask


def encode_bbox_to_mask(np_mask, material_sample):
    tlc = material_sample.bbox_top_left_corner_abs
    trc = material_sample.bbox_top_right_corner_abs
    brc = material_sample.bbox_bottom_right_corner_abs
    blc = material_sample.bbox_bottom_left_corner_abs

    tlc_enc_val = 255
    trc_enc_val = 254
    brc_enc_val = 253
    blc_enc_val = 252

    np_mask_layer = None

    if len(np_mask.shape) == 3:
        np_mask_layer = np_mask[:, :, 0]
    else:
        np_mask_layer = np_mask

    orig_vals = dict()
    orig_vals[tlc_enc_val] = np_mask_layer[tlc[0], tlc[1]]
    orig_vals[trc_enc_val] = np_mask_layer[trc[0], trc[1]]
    orig_vals[brc_enc_val] = np_mask_layer[brc[0], brc[1]]
    orig_vals[blc_enc_val] = np_mask_layer[blc[0], blc[1]]

    np_mask_layer[tlc[0], tlc[1]] = tlc_enc_val
    np_mask_layer[trc[0], trc[1]] = trc_enc_val
    np_mask_layer[brc[0], brc[1]] = brc_enc_val
    np_mask_layer[blc[0], blc[1]] = blc_enc_val

    return np_mask, orig_vals


def decode_bbox_from_mask(np_mask, orig_vals):

    corner_enc_vals = [255, 254, 253, 252]
    np_mask_layer = None

    if len(np_mask.shape) == 3:
        np_mask_layer = np_mask[:, :, 0]
    else:
        np_mask_layer = np_mask

    corners = np.argwhere(np.isin(np_mask_layer, corner_enc_vals))

    # Set the original values back
    if orig_vals is not None:
        for i in range(len(corners)):
            np_mask_layer[corners[i][0], corners[i][1]] = orig_vals[np_mask_layer[corners[i][0], corners[i][1]]]


    # If we could not recover enough corners, return the restored mask and a None
    # as the bounding box
    num_recovered_corners = len(corners)

    if num_recovered_corners < 2:
        return np_mask, None

    # It is possible that the interpolation has produced additional
    # corner color values. Take the minimum and maximum if the difference
    # in the coordinates is 2 pixels
    y_coords, x_coords = zip(*corners)
    y_min = min(y_coords)
    y_max = max(y_coords)
    x_min = min(x_coords)
    x_max = max(x_coords)

    y_diff = y_max-y_min
    x_diff = x_max-x_min

    if y_diff < 2 or x_diff < 2:
        return np_mask, None

    tlc = (y_min, x_min)
    trc = (y_min, x_max)
    brc = (y_max, x_max)
    blc = (y_max, x_min)

    return np_mask, (tlc, trc, brc, blc)


def process_segmentation_photo_mask_pair(np_photo,
                                         np_mask,
                                         material_sample,
                                         crop_shape,
                                         resize_shape,
                                         photo_cval,
                                         mask_cval,
                                         use_data_augmentation=False,
                                         data_augmentation_params=None,
                                         img_data_format='channels_last',
                                         div2_constraint=4,
                                         retry_crops=True):
    # type: (np.array, np.array, MaterialSample, tuple[int], tuple[int], np.array, np.array, bool, DataAugmentationParameters, str, int, bool) -> (np.array, np.array)

    """
    Applies crop and data augmentation to two numpy arrays representing the photo and
    the respective segmentation mask. The photos are not normalized to range [-1,1]
    within the function.

    # Arguments
        :param np_photo: the photo as a numpy array
        :param np_mask: the mask as a numpy array must have same spatial dimensions (HxW) as np_photo,
        :param material_sample: the material sample
        :param crop_shape: size of the crop or, None if no cropping should be applied
        :param resize_shape: size of the resized images, None if no resizing should be applied
        :param photo_cval: fill color value for photos [0,255]
        :param mask_cval: fill color value for masks [0,255]
        :param use_data_augmentation: should data augmentation be used
        :param data_augmentation_params: parameters for data augmentation
        :param img_data_format: format of the image data 'channels_first' or 'channels_last'
        :param div2_constraint: divisibility constraint
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
        np_photo = image_utils.np_scale_image_with_padding(np_photo, shape=resize_shape, cval=photo_cval, interp='bilinear')
        np_mask = image_utils.np_scale_image_with_padding(np_mask, shape=resize_shape, cval=mask_cval, interp='nearest')

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
    bbox = None

    if use_data_augmentation and np.random.random() <= data_augmentation_params.augmentation_probability:

        orig_vals, np_orig_photo, np_orig_mask = None, None, None

        if material_sample is not None:
            np_orig_photo = np.array(np_photo, copy=True)
            np_orig_mask = np.array(np_mask, copy=True)
            np_mask, orig_vals = encode_bbox_to_mask(np_mask, material_sample)

        np_photo, np_mask = image_utils.np_apply_random_transform(images=[np_photo, np_mask],
                                                                  cvals=[photo_cval, mask_cval],
                                                                  fill_mode=data_augmentation_params.fill_mode,
                                                                  img_data_format=img_data_format,
                                                                  rotation_range=data_augmentation_params.rotation_range,
                                                                  zoom_range=data_augmentation_params.zoom_range,
                                                                  width_shift_range=data_augmentation_params.width_shift_range,
                                                                  height_shift_range=data_augmentation_params.height_shift_range,
                                                                  channel_shift_ranges=[data_augmentation_params.channel_shift_range],
                                                                  horizontal_flip=data_augmentation_params.horizontal_flip,
                                                                  vertical_flip=data_augmentation_params.vertical_flip)

        if material_sample is not None:
            np_mask, bbox = decode_bbox_from_mask(np_mask, orig_vals)

            # If we could not decode from the augmented photo, skip the augmentation and
            # default to the original bbox from the material sample
            if bbox is None:
                bbox = (material_sample.bbox_top_left_corner_abs, material_sample.bbox_top_right_corner_abs, material_sample.bbox_bottom_right_corner_abs, material_sample.bbox_bottom_left_corner_abs)
                np_photo = np_orig_photo
                np_mask = np_orig_mask

    # If a crop size is given: take a random crop of both the image and the mask
    if crop_shape is not None:
        if dataset_utils.count_trailing_zeroes(crop_shape[0]) < div2_constraint or \
                        dataset_utils.count_trailing_zeroes(crop_shape[1]) < div2_constraint:
            raise ValueError('The crop size does not satisfy the div2 constraint of {}'.format(div2_constraint))

        # If we don't have a bounding box as a hint for cropping - take random crops
        if bbox is None:
            # Re-attempt crops if the crops end up getting only background
            # i.e. black pixels
            attempts = 5

            while attempts > 0:
                x1y1, x2y2 = image_utils.np_get_random_crop_area(np_mask, crop_shape[1], crop_shape[0])
                mask_crop = image_utils.np_crop_image(np_mask, x1y1[0], x1y1[1], x2y2[0], x2y2[1])

                # If the mask crop is only background (all R channel is zero) - try another crop
                # until attempts are exhausted
                if np.max(mask_crop[:, :, 0]) == 0 and attempts-1 != 0 and retry_crops:
                    attempts -= 1
                    continue

                np_mask = mask_crop
                np_photo = image_utils.np_crop_image(np_photo, x1y1[0], x1y1[1], x2y2[0], x2y2[1])
                break
        # Use the bounding box information to take a targeted crop
        else:
            tlc, trc, brc, blc = bbox
            crop_height = crop_shape[0]
            crop_width = crop_shape[1]
            bbox_ymin = tlc[0]
            bbox_ymax = blc[0]
            bbox_xmin = tlc[1]
            bbox_xmax = trc[1]
            bbox_height = bbox_ymax-bbox_ymin
            bbox_width = bbox_xmax-bbox_xmin
            #bbox_center = ((bbox_ymin + bbox_ymax) / 2, (bbox_xmin + bbox_xmax) / 2)
            height_diff = abs(bbox_height - crop_height)
            width_diff = abs(bbox_width - crop_width)

            # If the crop can fit the whole material sample within it
            if bbox_height < crop_height and bbox_width < crop_width:
                crop_ymin = bbox_ymin - np.random.randint(0, min(height_diff, bbox_ymin+1))
                crop_ymax = crop_ymin + crop_height
                crop_xmin = bbox_xmin - np.random.randint(0, min(width_diff, bbox_xmin+1))
                crop_xmax = crop_xmin + crop_width

            # If the crop can't fit the whole material sample within it
            else:
                crop_ymin = bbox_ymin + np.random.randint(0, height_diff+1)
                crop_ymax = crop_ymin + crop_height
                crop_xmin = bbox_xmin + np.random.randint(0, width_diff+1)
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

    if img_height_div2 < div2_constraint or img_width_div2 < div2_constraint:

        # The photo needs to be filled with the photo_cval color and masks with the mask cval color
        padded_shape = dataset_utils.get_required_image_dimensions(np_photo.shape, div2_constraint)
        np_photo = image_utils.np_pad_image_to_shape(np_photo, padded_shape, photo_cval)
        np_mask = image_utils.np_pad_image_to_shape(np_mask, padded_shape, mask_cval)

    return np_photo, np_mask


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
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 channel_shift_range=None,
                 horizontal_flip=False,
                 vertical_flip=False,
                 fill_mode='constant'):
        """
        # Arguments
            :param augmentation_probability: probability with which to apply random augmentations
            :param rotation_range: range of random rotations
            :param zoom_range: range of random zooms
            :param width_shift_range: fraction of total width [0, 1]
            :param height_shift_range: fraction of total height [0, 1]
            :param channel_shift_range: channel shift range as a float, enables shifting channels between [-val, val]
            :param horizontal_flip: should horizontal flips be applied
            :param vertical_flip: should vertical flips be applied
            :param fill_mode: how should we fill overgrown areas
        # Returns
            :return: A new instance of DataAugmentationParameters
        """

        self.augmentation_probability = augmentation_probability
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.channel_shift_range = channel_shift_range
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
                 use_material_samples=False):

        self.material_class_information = material_class_information
        self.num_color_channels = num_color_channels
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


#######################################
# ITERATOR
#######################################

class IterationMode(Enum):
    UNIFORM = 0,  # Sample each material class uniformly wrt. number of samples within epoch
    UNIQUE = 1   # Iterate through all the unique samples once within epoch


class DataSetIterator(object):

    __metaclass__ = ABCMeta

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = min(batch_size, n)    # The batch size could in theory be bigger than the data set size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0

    def reset(self):
        self.batch_index = 0
        self.total_batches_seen = 0

    @abstractmethod
    def get_next_batch(self):
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

    @abstractproperty
    def num_steps_per_epoch(self):
        pass


class FileDataSetIterator(DataSetIterator):
    """
    A class for iterating over a data set in batches.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        # type: (int, int, bool, int) -> None

        """
        # Arguments
            :param n: Integer, total number of samples in the dataset to loop over.
            :param batch_size: Integer, size of a batch.
            :param shuffle: Boolean, whether to shuffle the data between epochs.
            :param seed: Random seeding for data shuffling.
        # Returns
            Nothing
        """
        super(FileDataSetIterator, self).__init__(n=n, batch_size=batch_size, shuffle=shuffle, seed=seed)
        self.index_array = np.arange(self.n)

    def get_next_batch(self):
        # type: () -> (np.array[int], int, int)

        super(FileDataSetIterator, self).get_next_batch()

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

        self.total_batches_seen += 1

        return self.index_array[current_index: current_index + current_batch_size], current_index, current_batch_size

    @property
    def num_steps_per_epoch(self):
        return dataset_utils.get_number_of_batches(self.n, self.batch_size)


class MaterialSampleDataSetIterator(DataSetIterator):
    """
    A class for iterating randomly through MaterialSamples for a data set in batches.
    """

    def __init__(self, material_samples, batch_size, shuffle, seed, iter_mode=IterationMode.UNIFORM):
        # type: (list[list[MaterialSample]], int, bool, int, bool) -> None

        self._num_unique_material_samples = sum(len(material_category) for material_category in material_samples)
        super(MaterialSampleDataSetIterator, self).__init__(n=self._num_unique_material_samples, batch_size=batch_size, shuffle=shuffle, seed=seed)

        # Calculate uniform probabilities for all classes that have non zero samples
        self.iter_mode = iter_mode
        self._class_probabilities = [0.0] * len(material_samples)
        self._num_non_zero_classes = sum(1 for material_category in material_samples if len(material_category) > 0)
        self._num_samples_in_biggest_material_category = max(len(material_category) for material_category in material_samples)
        self._num_samples_in_smallest_material_category = min(len(material_category) for material_category in material_samples)

        # Build index lists for the different material samples
        self._material_samples = []

        for material_category in material_samples:
            if not shuffle:
                self._material_samples.append(np.arange(len(material_category)))
            else:
                self._material_samples.append(np.random.permutation(len(material_category)))

        # Build a flattened list of all the material samples (for regular iteration)
        self._material_samples_flattened = []

        for i in range(len(self._material_samples)):
            for j in range(len(self._material_samples[i])):
                self._material_samples_flattened.append((i, j))

        for i in range(len(material_samples)):
            num_samples_in_category = len(material_samples[i])

            if num_samples_in_category > 0:
                self._class_probabilities[i] = 1.0 / self._num_non_zero_classes
            else:
                # Zero is assumed as the background class and should/can have zero instances
                if i != 0:
                    warnings.warn('Material class {} has 0 material samples'.format(i), RuntimeWarning)

                self._class_probabilities[i] = 0.0

        # Keep track of the current sample in each material category
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

        if self.iter_mode == IterationMode.UNIQUE:
            return self._get_next_batch_unique()
        elif self.iter_mode == IterationMode.UNIFORM:
            return self._get_next_batch_uniform()

        raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

    def _get_next_batch_uniform(self):
        # Class 0 is assumed to be background so classes will be [1,num_classes-1]
        sample_classes = np.random.choice(a=self.num_material_classes, size=self.batch_size, p=self._class_probabilities)
        ret = []

        for sample_category_idx in sample_classes:
            internal_sample_idx = self._current_samples[sample_category_idx]
            sample_idx = self._material_samples[sample_category_idx][internal_sample_idx]
            ret.append((sample_category_idx, sample_idx))

            # Keep track of the used samples in each category
            self._current_samples[sample_category_idx] += 1

            # If all of the samples in the category have been used, zero out the
            # index for the category and shuffle the category list if shuffle is enabled
            if self._current_samples[sample_category_idx] >= len(self._material_samples[sample_category_idx]):
                self._current_samples[sample_category_idx] = 0

                if self.shuffle:
                    self._material_samples[sample_category_idx] = np.random.permutation(len(self._material_samples[sample_category_idx]))
                else:
                    self._material_samples[sample_category_idx] = np.arange(len(self._material_samples[sample_category_idx]))

        # Keep track of how many times we have gone "through all the samples"
        n_samples = self._num_samples_in_biggest_material_category*self._num_non_zero_classes
        current_index = (self.batch_index * self.batch_size) % n_samples
        current_batch_size = len(ret)

        if n_samples > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        self.total_batches_seen += 1

        return ret, current_index, current_batch_size

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

        self.total_batches_seen += 1

        return self._material_samples_flattened[current_index: current_index + current_batch_size], current_index, current_batch_size

    @property
    def num_steps_per_epoch(self):
        if self.iter_mode == IterationMode.UNIQUE:
            return dataset_utils.get_number_of_batches(self._num_unique_material_samples, self.batch_size)
        elif self.iter_mode == IterationMode.UNIFORM:
            # If all classes are sampled uniformly, we have been through all the samples in the data
            # on average after we have gone through all the samples in the largest class
            return dataset_utils.get_number_of_batches(self._num_samples_in_biggest_material_category * self._num_non_zero_classes, self.batch_size)

        raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

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
        # type: (DataGeneratorParameters) -> None

        self.lock = threading.Lock()

        # Unwrap DataGeneratorParameters to member variables
        self.material_class_information = params.material_class_information
        self.num_color_channels = params.num_color_channels
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

        # Other member variables
        self.img_data_format = K.image_data_format()

        # Ensure the per_channel_mean is a numpy tensor
        if self.per_channel_mean is not None:
            self.per_channel_mean = np.array(self.per_channel_mean)

        # Ensure per_channel_stddev is a numpy tensor
        if self.per_channel_stddev is not None:
            self.per_channel_stddev = np.array(self.per_channel_stddev)

        # Calculate missing per-channel mean if necessary
        if self.use_per_channel_mean_normalization and (self.per_channel_mean is None or len(self.per_channel_mean) < self.num_color_channels):
            self.per_channel_mean = dataset_utils.calculate_per_channel_mean(self.get_all_photos(), self.num_color_channels)
            print 'DataGenerator: Using per-channel mean: {}'.format(self.per_channel_mean)

        # Calculate missing per-channel stddev if necessary
        if self.use_per_channel_stddev_normalization and (self.per_channel_stddev is None or len(self.per_channel_stddev) < self.num_color_channels):
            self.per_channel_stddev = dataset_utils.calculate_per_channel_stddev(self.get_all_photos(), self.per_channel_mean, self.num_color_channels)
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
    def get_all_masks(self):
        # type: () -> list[ImageFile]

        """
        Returns all the mask ImageFile instances as a list.

        # Arguments
            None
        # Returns
            :return: all the mask ImageFiles as a list
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


#######################################
# SEGMENTATION DATA GENERATOR
#######################################

class SegmentationDataGenerator(DataGenerator):
    """
    DataGenerator which provides supervised segmentation data. Produces batches
    of matching image segmentation mask pairs.
    """

    def __init__(self, labeled_data_set, num_labeled_per_batch, params):
        # type: (LabeledImageDataSet, int, DataGeneratorParameters) -> None

        """
        # Arguments
            :param labeled_data_set: LabeledImageDataSet instance of the data
            :param num_labeled_per_batch: number of labeled data set samples per batch
            :param params: DataGeneratorParams instance for parameters
        """

        self.labeled_data_set = labeled_data_set
        self.num_labeled_per_batch = num_labeled_per_batch

        super(SegmentationDataGenerator, self).__init__(params)

        if self.use_material_samples:

            if self.labeled_data_set.material_samples is None or len(self.labeled_data_set.material_samples) == 0:
                raise ValueError('Use material samples is true, but labeled data set does not contain material samples')

            self.labeled_data_iterator = MaterialSampleDataSetIterator(
                material_samples=self.labeled_data_set.material_samples,
                batch_size=num_labeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed)
        else:
            self.labeled_data_iterator = FileDataSetIterator(
                n=self.labeled_data_set.size,
                batch_size=num_labeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed)

    def get_all_photos(self):
        # type: () -> list[ImageFile]

        """
        Returns all the photo ImageFile instances as list.

        # Arguments
            None
        # Returns
            :return: all the photo ImageFiles as a list
        """
        photos = []
        photos += self.labeled_data_set.photo_image_set.image_files
        return photos

    def get_all_masks(self):
        # type: () -> list[ImageFile]

        """
        Returns all the mask ImageFile instances as a list.

        # Arguments
            None
        # Returns
            :return: all the mask ImageFiles as a list
        """
        masks = []
        masks += self.labeled_data_set.mask_image_set.image_files
        return masks

    def next(self):
        # type: () -> (np.array, np.array)

        """
        Yields batches of data endlessly.

        # Arguments
            None
        # Returns
            :return: batch of input image, segmentation mask data as a tuple (X,Y)
        """

        with self.lock:
            labeled_index_array, labeled_current_index, labeled_current_batch_size = self.labeled_data_iterator.get_next_batch()

        if self.use_material_samples:
            batch_files, material_samples = self.labeled_data_set.get_files_and_material_samples(labeled_index_array)
        else:
            batch_files, material_samples = self.labeled_data_set.get_indices(labeled_index_array), None

        # Parallel processing of the files in this batch
        data = [get_labeled_segmentation_data_pair(
                photo_mask_pair=batch_files[i],
                material_sample=material_samples[i] if material_samples is not None else None,
                num_color_channels=self.num_color_channels,
                crop_shape=self.crop_shape,
                resize_shape=self.resize_shape,
                material_class_information=self.material_class_information,
                photo_cval=self.photo_cval,
                mask_cval=self.mask_cval,
                use_data_augmentation=self.use_data_augmentation,
                data_augmentation_params=self.data_augmentation_params,
                img_data_format=self.img_data_format,
                div2_constraint=4,
                mask_type='one_hot') for i in range(len(batch_files))]

        # Note: all the examples in the batch have to have the same dimensions
        X, Y = zip(*data)
        X, Y = np.array(X), np.array(Y)

        # Normalize the photo batch data
        X = dataset_utils \
            .normalize_batch(X,
                             self.per_channel_mean if self.use_per_channel_mean_normalization else None,
                             self.per_channel_stddev if self.use_per_channel_stddev_normalization else None)

        return X, Y

    @property
    def num_steps_per_epoch(self):
        """
        Returns the number of batches (steps) per epoch.

        # Arguments
            None
        # Returns
            :return: Number of steps per epoch
        """
        return self.labeled_data_iterator.num_steps_per_epoch


################################################
# SEMI SUPERVISED SEGMENTATION DATA GENERATOR
################################################


class SemisupervisedSegmentationDataGenerator(DataGenerator):

    def __init__(self,
                 labeled_data_set,
                 unlabeled_data_set,
                 num_labeled_per_batch,
                 num_unlabeled_per_batch,
                 params,
                 class_weights=None,
                 label_generation_function=None):
        # type: (LabeledImageDataSet, UnlabeledImageDataSet, int, int, DataGeneratorParameters, function[[np.array[np.float32]], np.array]) -> None

        """
        # Arguments
            :param labeled_data_set:
            :param unlabeled_data_set:
            :param num_labeled_per_batch: number of labeled images per batch
            :param num_unlabeled_per_batch: number of unlabeled images per batch
            :param params: DataGeneratorParameters object
            :param class_weights: class weights
            :param label_generation_function:
        """

        self.labeled_data_set = labeled_data_set
        self.unlabeled_data_set = unlabeled_data_set

        self.num_labeled_per_batch = num_labeled_per_batch
        self.num_unlabeled_per_batch = num_unlabeled_per_batch

        super(SemisupervisedSegmentationDataGenerator, self).__init__(params)

        if self.use_material_samples:
            if self.labeled_data_set.material_samples is None or len(self.labeled_data_set.material_samples) == 0:
                raise ValueError('Use material samples is true, but labeled data set does not contain material samples')

            self.labeled_data_iterator = MaterialSampleDataSetIterator(
                material_samples=self.labeled_data_set.material_samples,
                batch_size=num_unlabeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed)

        else:
            self.labeled_data_iterator = FileDataSetIterator(
                n=self.labeled_data_set.size,
                batch_size=num_labeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed)

        self.unlabeled_data_iterator = None

        if unlabeled_data_set is not None and unlabeled_data_set.size > 0:
            self.unlabeled_data_iterator = FileDataSetIterator(
                n=self.unlabeled_data_set.size,
                batch_size=num_unlabeled_per_batch,
                shuffle=self.shuffle_data_after_epoch,
                seed=self.random_seed)

        if labeled_data_set is None:
            raise ValueError('SemisupervisedSegmentationDataGenerator does not support empty labeled data set')

        if label_generation_function is None:
            self.label_generation_function = SemisupervisedSegmentationDataGenerator.default_label_generator_for_unlabeled_photos
        else:
            self.label_generation_function = label_generation_function

        if class_weights is None:
            raise ValueError('Class weights is None. Use a numpy array of ones instead of None')

        self.class_weights = class_weights

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

        if self.unlabeled_data_set is not None:
            photos += self.unlabeled_data_set.photo_image_set.image_files

        return photos

    def get_all_masks(self):
        # type: () -> list[ImageFile]

        """
        Returns all the mask file paths as a list.

        # Arguments
            None
        # Returns
            :return: all the mask file paths as a list
        """
        masks = []
        masks += self.labeled_data_set.mask_image_set.image_files
        return masks

    def has_unlabeled_data(self):
        # type: () -> bool

        """
        Returns whether the generator has unlabeled data.

        # Arguments
            None
        # Returns
            :return: true if there is unlabeled data otherwise false
        """
        return self.unlabeled_data_set is not None and self.unlabeled_data_set.size > 0 and self.unlabeled_data_iterator is not None

    def get_labeled_batch_data(self, index_array):
        # type: (np.array[int]) -> (list[np.array], list[np.array])

        """
        # Arguments
            :param index_array: indices of the labeled data
        # Returns
            :return: labeled data as three lists: (X, Y, WEIGHTS)
        """

        if self.use_material_samples:
            labeled_batch_files, material_samples = self.labeled_data_set.get_files_and_material_samples(index_array)
        else:
            labeled_batch_files, material_samples = self.labeled_data_set.get_indices(index_array), None

        # Process the labeled files for this batch
        labeled_data = [get_labeled_segmentation_data_pair(
                photo_mask_pair=labeled_batch_files[i],
                material_sample=material_samples[i] if material_samples is not None else None,
                num_color_channels=self.num_color_channels,
                crop_shape=self.crop_shape,
                resize_shape=self.resize_shape,
                material_class_information=self.material_class_information,
                photo_cval=self.photo_cval,
                mask_cval=self.mask_cval,
                use_data_augmentation=self.use_data_augmentation,
                data_augmentation_params=self.data_augmentation_params,
                img_data_format=self.img_data_format,
                div2_constraint=4,
                mask_type='index') for i in range(len(labeled_batch_files))]

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

    def get_unlabeled_batch_data(self, index_array):
        # type: (np.array[int]) -> (list[np.array], list[np.array], list[np.array])

        """
        # Arguments
            :param index_array: indices of the unlabeled data
        # Returns
            :return: unlabeled data as three lists: (X, Y, WEIGHTS)
        """

        # If we don't have unlabeled data return two empty lists
        if not self.has_unlabeled_data():
            return [], [], []

        unlabeled_batch_files = self.unlabeled_data_set.get_indices(index_array)

        # Process the unlabeled data pairs (take crops, apply data augmentation, etc).
        unlabeled_data = [get_unlabeled_segmentation_data_pair(
            photo=photo,
            label_generation_function=self.label_generation_function,
            num_color_channels=self.num_color_channels,
            crop_shape=self.crop_shape,
            resize_shape=self.resize_shape,
            photo_cval=self.photo_cval,
            mask_cval=[0],
            use_data_augmentation=self.use_data_augmentation,
            data_augmentation_params=self.data_augmentation_params,
            img_data_format=self.img_data_format,
            div2_constraint=4) for photo in unlabeled_batch_files]

        X_unlabeled, Y_unlabeled = zip(*unlabeled_data)
        W_unlabeled = []

        for y in Y_unlabeled:
            W_unlabeled.append(np.ones_like(y, dtype=np.float32))

        return list(X_unlabeled), list(Y_unlabeled), W_unlabeled

    def next(self):
        # type: (int, int, tuple[int]) -> (list[np.array], np.array)

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
            labeled_index_array, labeled_current_index, labeled_current_batch_size = self.labeled_data_iterator.get_next_batch()
            unlabeled_index_array, unlabeled_current_index, unlabeled_current_batch_size = None, None, None

            if self.has_unlabeled_data():
                unlabeled_index_array, unlabeled_current_index, unlabeled_current_batch_size = self.unlabeled_data_iterator.get_next_batch()

        X, Y, W = self.get_labeled_batch_data(labeled_index_array)
        X_unlabeled, Y_unlabeled, W_unlabeled = self.get_unlabeled_batch_data(unlabeled_index_array)
        X = X + X_unlabeled
        Y = Y + Y_unlabeled
        W = W + W_unlabeled

        num_unlabeled_samples_in_batch = len(X_unlabeled)
        num_samples_in_batch = len(X)

        # Debug: Write images
        #for i in range(len(X)):
        #    from keras.preprocessing.image import array_to_img
        #    _debug_photo = array_to_img(X[i])
        #    _debug_photo.save('./photos/debug-photos/{}_{}_photo.jpg'.format(labeled_current_index, i), format='JPEG')
        #    _debug_mask = array_to_img(Y[i][:, :, np.newaxis]*255)
        #    _debug_mask.save('./photos/debug-photos/{}_{}_mask.png'.format(labeled_current_index, i), format='PNG')
        # End of: debug

        # Cast the lists to numpy arrays
        X, Y, W = np.array(X), np.array(Y), np.array(W)

        # Normalize the photo batch data
        X = dataset_utils\
            .normalize_batch(X,
                             self.per_channel_mean if self.use_per_channel_mean_normalization else None,
                             self.per_channel_stddev if self.use_per_channel_stddev_normalization else None)

        # The dimensions of the number of unlabeled in the batch must match with batch dimension
        num_unlabeled = np.ones(shape=[num_samples_in_batch], dtype=np.int32) * num_unlabeled_samples_in_batch

        # Generate a dummy output for the dummy loss function and yield a batch of data
        dummy_output = np.zeros(shape=[num_samples_in_batch])

        batch_data = [X, Y, W, num_unlabeled]

        if X.shape[0] != Y.shape[0] or X.shape[0] != W.shape[0] or X.shape[0] != num_unlabeled.shape[0]:
            print 'Unmatching input first dimensions: {}, {}, {}, {}'.format(X.shape[0], Y.shape[0], W.shape[0], num_unlabeled.shape[0])

        return batch_data, dummy_output

    @property
    def num_steps_per_epoch(self):
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

#
# # Dirty trick to make pickling possible for member functions, see:
# # https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma
# def unwrap_get_data_pair(arg, **kwarg):
#     return ClassificationDataGenerator.get_data_pair(*arg, **kwarg)
#
#
# class ClassificationDataGenerator(object):
#     """
#     DataGenerator which provides supervised classification data. Produces batches
#     of matching image one-hot-encoded vector pairs.
#     """
#
#     def __init__(self,
#                  path_to_images_folder,
#                  per_channel_mean,
#                  path_to_labels_file,
#                  path_to_categories_file,
#                  num_channels,
#                  random_seed=None,
#                  verbose=False,
#                  shuffle_data_after_epoch=True):
#
#         self.path_to_images_folder = path_to_images_folder
#         self.per_channel_mean = per_channel_mean
#         self.path_to_labels_file = path_to_labels_file
#         self.path_to_categories_file = path_to_categories_file
#         self.num_channels = num_channels
#         self.verbose = verbose
#         self.shuffle_data_after_epoch = shuffle_data_after_epoch
#
#         # Read categories file and build the categories dictionary
#         # category name -> category index
#         self.categories = {}
#
#         with open(path_to_categories_file, 'r') as f:
#             lines = f.readlines()
#             # Remove whitespace characters like `\n` at the end of each line
#             lines = [x.strip() for x in lines]
#
#             for idx, category in enumerate(lines):
#                 self.categories[category] = idx
#
#         self.num_categories = len(self.categories)
#
#         if self.verbose:
#             print 'Found {} categories: {}'.format(self.num_categories, self.categories)
#
#         # Read the image file paths into an array
#         with open(path_to_labels_file, 'r') as f:
#             self.files = f.readlines()
#             # Remove whitespace characters like `\n` at the end of each line
#             self.files = [x.strip() for x in self.files]
#
#         self.num_samples = len(self.files)
#
#         if self.verbose:
#             print 'Found {} samples'.format(self.num_samples)
#
#         if random_seed is not None:
#             np.random.seed(random_seed)
#             random.seed()
#
#     def get_data_pair(self, f):
#         # type: (str) -> (np.array, np.array)
#
#         """
#         Creates a data pair of image and one-hot-encoded category vector from a MINC
#         photo path.
#
#         # Arguments
#             :param f: path to MINC photo
#         # Returns
#             :return: a tuple of numpy arrays (image, one-hot-encoded category vector)
#         """
#
#         # The path is always /images/<category>/<filename>
#         file_parts = f.split('/')
#         category = file_parts[1]
#         filename = file_parts[2]
#
#         # Build the complete path
#         file_path = os.path.join(self.path_to_images_folder, category, filename)
#
#         # Load the image
#         img = dataset_utils.load_n_channel_image(file_path, self.num_channels)
#         x = img_to_array(img)
#
#         # Create the one-hot-vector for the classification
#         category_index = self.categories[category]
#         y = np.zeros(self.num_categories)
#         y[category_index] = 1.0
#
#         return x, y
#
#     @threadsafe
#     def get_flow(self, batch_size):
#
#         dataset_size = len(self.files)
#         num_batches = dataset_utils.get_number_of_batches(dataset_size, batch_size)
#         n_jobs = dataset_utils.get_number_of_parallel_jobs()
#
#         while True:
#             if self.shuffle_data_after_epoch:
#                 # Shuffle the files
#                 random.shuffle(self.files)
#
#             for i in range(0, num_batches):
#                 # Get the files for this batch
#                 batch_range = dataset_utils.get_batch_index_range(dataset_size, batch_size, i)
#                 files = self.files[batch_range[0]:batch_range[1]]
#
#                 data = Parallel(n_jobs=n_jobs, backend='threading')(
#                     delayed(unwrap_get_data_pair)((self, f)) for f in files)
#
#                 X, Y = zip(*data)
#                 X, Y = dataset_utils.normalize_batch(np.array(X)), np.array(Y)
#
#                 yield X, Y
