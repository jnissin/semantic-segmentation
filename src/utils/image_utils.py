# coding=utf-8

import numpy as np
import copy
import skimage.transform as skitransform

from enum import Enum

from skimage.exposure import adjust_gamma
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, find_boundaries
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.transform import SimilarityTransform

from keras.preprocessing.image import array_to_img, img_to_array, flip_axis


class ImageInterpolation(Enum):
    NEAREST = 0
    BILINEAR = 1
    BIQUADRATIC = 2
    BICUBIC = 3
    BIQUARTIC = 4
    BIQUINTIC = 5


class ImageValidationErrorType(Enum):
    NONE = 0
    INF_VALUES = 1
    OUT_OF_RANGE_VALUES = 2
    DIMENSION = 3
    DTYPE= 4


class ImageValidationError(Exception):
    def __init__(self, message, error):

        # Call the base class constructor with the parameters it needs
        super(ImageValidationError, self).__init__(message)
        self.error = error


class ImageTransform:

    def __init__(self,
                 image_height,
                 image_width,
                 transform,
                 horizontal_flip,
                 vertical_flip):
        # type: (int, int, SimilarityTransform, bool, bool) -> None

        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def transform_normalized_coordinates(self, coordinates):
        # type: (np.ndarray) -> np.ndarray

        """
        Applies the same operations to the coordinate [x,y] as were applied in the ImageTransform,
        in the same order: transform, horizontal flip, vertical flip. The coordinates should be
        in range [0,1].

        # Arguments
            :param coordinates: the coordinate as a numpy array [x,y]
        # Returns
            :return: the transformed coordinate [x',y']
        """

        p = np.array(coordinates)
        p_rank = len(p.shape)

        if not 1 <= p_rank <= 2:
            raise ValueError('The coordinates must be either of rank 1 or 2')

        # If we have N coordinates
        if p_rank == 1:
            p[0] *= self.image_width
            p[1] *= self.image_height
        elif p_rank == 2:
            p[:, 0] *= self.image_width
            p[:, 1] *= self.image_height

        p = self.transform_coordinates(p)

        if p_rank == 1:
            p[0] /= self.image_width
            p[1] /= self.image_height
        elif p_rank == 2:
            p[:, 0] /= self.image_width
            p[:, 1] /= self.image_height

        return p

    def transform_coordinates(self, coordinate):
        # type: (np.ndarray) -> np.ndarray

        """
        Applies the same operations to the coordinate(s) [x,y] as were applied in the ImageTransform,
        in the same order: transform, horizontal flip, vertical flip. The coordinates should be
        in range [0, IMG_HEIGHT] and [0, IMG_WIDTH].

        # Arguments
            :param coordinate: the coordinate as a numpy array [x,y]
        # Returns
            :return: the transformed coordinate [x',y']
        """

        p = np.array(coordinate).astype(np.float32)

        if not 1 <= p.ndim <= 2:
            raise ValueError('The coordinates must be either of rank 1 or 2: [2] or [N, 2]')

        if self.horizontal_flip:
            if p.ndim == 1:
                p[0] = self.image_width - p[0]
            elif p.ndim == 2:
                p[:, 0] = self.image_width - p[:, 0]

        if self.vertical_flip:
            if p.ndim == 1:
                p[1] = self.image_height - p[1]
            elif p.ndim == 2:
                p[:, 1] = self.image_height - p[:, 1]

        p = skitransform.matrix_transform(p, self.transform.params)

        # The matrix_transform always returns an ndarray of [N,2] if we had rank 1 squeeze the extra dimension
        if p.ndim == 1:
            p = np.squeeze(p)

        return p


##############################################
# NUMPY IMAGE FUNCTIONS
##############################################

def np_check_image_properties(np_img, min_val=0.0, max_val=255.0, height=None, width=None, dtype=None):
    # type: (np.ndarray, float, float, int, int, np.dtype) -> None

    """
    Checks the image properties and raises ImageValidationError if assumptions do not match the image.
    The exact problem can be found by examining the error field of the raised ImageValidationError.

    # Arguments
        :param np_img: the image
        :param min_val: assumed minimum value found in the image
        :param max_val: assumed maximum value found in the image
        :param height: assumed height of the image, None if not checking
        :param width: assumed width of the image, None if not checking
        :param dtype: assumed dtype of the image, None if not checking

    # Returns
        :return: ImagePropertyError describing the error or ImagePropertyError.NONE if no error
    """
    if dtype is not None:
        if np_img.dtype != dtype:
            raise ImageValidationError('Found invalid dtype in image, assumed {}, got: {}'
                                       .format(dtype, np_img.dtype), ImageValidationErrorType.DTYPE)

    if height is not None:
        if np_img.shape[0] != height:
            raise ImageValidationError('Found invalid height in image, assumed {}, got: {}'
                                       .format(height, np_img.shape[0]), ImageValidationErrorType.DIMENSION)

    if width is not None:
        if np_img.shape[1] != width:
            raise ImageValidationError('Found invalid width in image, assumed {}, got: {}'
                                       .format(width, np_img.shape[1]), ImageValidationErrorType.DIMENSION)

    if np.any(np.invert(np.isfinite(np_img))):
        raise ImageValidationError('Found inf values in image', ImageValidationErrorType.INF_VALUES)

    if min_val is not None:
        if np.min(np_img) < min_val:
            raise ImageValidationError('Found invalid min values in image, assumed: {}, got: {}'
                                       .format(min_val, np.min(np_img)), ImageValidationErrorType.OUT_OF_RANGE_VALUES)

    if max_val is not None:
        if np.max(np_img) > max_val:
            raise ImageValidationError('Found invalid max values in image, assumed: {}, got: {}'
                                       .format(max_val, np.max(np_img)), ImageValidationErrorType.OUT_OF_RANGE_VALUES)

    return ImageValidationErrorType.NONE


def np_apply_random_transform(images,
                              cvals,
                              fill_mode='constant',
                              interpolations=None,
                              transform_origin=None,
                              img_data_format='channels_last',
                              rotation_range=None,
                              zoom_range=None,
                              gamma_adjust_ranges=None,
                              width_shift_range=0.0,
                              height_shift_range=0.0,
                              channel_shift_ranges=None,
                              horizontal_flip=False,
                              vertical_flip=False):
    # type: (list[np.ndarray], list[np.ndarray], str, list[ImageInterpolation], np.array, str, np.ndarray, np.ndarray, list[np.ndarray], float, float, list[np.ndarray], bool, bool) -> (list[np.ndarray], ImageTransform)

    """
    Randomly augments, in the same way, a list of numpy images.

    # Arguments
        :param images: a list of 3D tensors, image colors in range [0,255]
        :param cvals: the fill values for the images, should be the same size as images list
        :param fill_mode: how to fill the image
        :param interpolations: a list of order of spline interpolations for each image or None if interpolation is not used for any images
        :param transform_origin: a custom transform origin for the transformations [y,x] in normalized [0,1] img coordinates, image center will be used if nothing given
        :param img_data_format: format of the image data (channels_last or channels_first)
        :param rotation_range: range of the rotations in degrees
        :param zoom_range: zoom range > 1 zoom in, < 1 zoom out
        :param gamma_adjust_range: list of gamma adjustment ranges
        :param width_shift_range: fraction of total width [0, 1]
        :param height_shift_range: fraction of total height [0, 1]
        :param channel_shift_ranges: a list of channel shift ranges for each image, must be shorter or same length as images list
        :param horizontal_flip: should horizontal flips be applied
        :param vertical_flip: should vertical flips be applied
    # Returns
        :return: Inputs (x, y) with the same random transform applied.
    """

    if images is None or len(images) == 0:
        return images

    # Figure out the correct axes according to image data format
    if img_data_format == 'channels_first':
        img_channel_axis = 0
        img_row_axis = 1
        img_col_axis = 2
    elif img_data_format == 'channels_last':
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2
    else:
        raise ValueError('Unknown image data format: {}'.format(img_data_format))

    # Make sure the images and fill values match
    if len(cvals) != len(images):
        raise ValueError('Unmatching image and cvalue array lengths: {} vs {}', len(cvals), len(images))

    for i in range(0, len(images)):
        if len(cvals[i]) != images[i].shape[img_channel_axis]:
            raise ValueError('Unmatching fill value dimensions for image element {}: {} vs {}'.format(i, len(cvals[i]), images[i].shape[img_channel_axis]))

    # Make sure the images have the same dimensions HxW
    img_width = images[0].shape[img_col_axis]
    img_height = images[0].shape[img_row_axis]
    img_dtype = images[0].dtype

    # Store the ImageTranform for return value
    img_transform = ImageTransform(image_height=img_height, image_width=img_width, transform=None, horizontal_flip=False, vertical_flip=False)

    # Make sure the dimensions match and the values are in the expected initial range [0, 255]
    for i in range(0, len(images)):
        np_check_image_properties(images[0], min_val=0.0, max_val=255.0, height=img_height, width=img_width, dtype=img_dtype)

    # Apply gamma adjustment to the image
    # Note: apply before transform to keep the possible cvalue always constant in the transformed images
    if gamma_adjust_ranges is not None:
        if len(gamma_adjust_ranges) > len(images):
            raise ValueError('Gamma adjustment ranges list is longer than the image list: {} vs {}'.format(len(gamma_adjust_ranges), len(images)))

        for i in range(0, len(gamma_adjust_ranges)):
            if gamma_adjust_ranges[i] is None:
                continue

            # We need to give the images as type uin8 to maintain the range [0, 255] transform to uint8 and back
            gamma = np.random.uniform(gamma_adjust_ranges[i][0], gamma_adjust_ranges[i][1])
            images[i] = adjust_gamma(images[i].astype(np.uint8), gamma=gamma).astype(img_dtype)

    # Apply random channel shifts
    # Note: apply before transform to keep the possible cvalue always constant in the transformed images
    if channel_shift_ranges is not None:
        if len(channel_shift_ranges) > len(images):
            raise ValueError('Channel shift ranges list is longer than the image list: {} vs {}'.format(len(channel_shift_ranges), len(images)))

        for i in range(0, len(channel_shift_ranges)):
            if channel_shift_ranges[i] is None:
                continue

            # Images are [0,255] color encoded, multiply intensity [0,1] by 255 to get the real shift intensity
            images[i] = np_random_channel_shift(images[i], intensity=channel_shift_ranges[i] * 255.0, min_c=0, max_c=255.0, channel_axis=img_channel_axis)

    # Apply at random a horizontal flip to the image
    if horizontal_flip:
        if np.random.random() < 0.5:
            for i in range(0, len(images)):
                images[i] = flip_axis(images[i], img_col_axis)
            img_transform.horizontal_flip = True

    # Apply at random a vertical flip to the image
    if vertical_flip:
        if np.random.random() < 0.5:
            for i in range(0, len(images)):
                images[i] = flip_axis(images[i], img_row_axis)
            img_transform.vertical_flip = True

    # Rotation
    if rotation_range:
        theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    else:
        theta = 0.0

    # Height shift
    if height_shift_range is not None and height_shift_range > 0.0:
        ty = np.random.uniform(-height_shift_range, height_shift_range) * images[0].shape[img_row_axis]
    else:
        ty = 0

    # Width shift
    if width_shift_range is not None and width_shift_range > 0.0:
        tx = np.random.uniform(-width_shift_range, width_shift_range) * images[0].shape[img_col_axis]
    else:
        tx = 0

    # Zoom
    if zoom_range is None or (zoom_range[0] == 1 and zoom_range[1] == 1):
        zoom = 1
    else:
        # Do not shear when zooming - i.e. assign same value to x and y.
        zoom = np.random.uniform(zoom_range[0], zoom_range[1])

    # Calculate necessary movement to shift the origin to the image center or
    # the given transform origin
    if transform_origin is not None:
        shift_y = img_height * transform_origin[0]
        shift_x = img_width * transform_origin[1]
    else:
        shift_y = img_height * 0.5
        shift_x = img_width * 0.5

    # Prepare transforms to shift the image origin to the image center
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

    # Build the translation, rotation and scale transforms
    tf_translate = SimilarityTransform(translation=[tx, ty])
    tf_rotate = SimilarityTransform(rotation=theta)
    tf_scale = SimilarityTransform(scale=zoom)

    # Build the final transform: (SHIFT)*S*R*T*(SHIFT_INV)
    tf_final = (tf_shift + (tf_scale + tf_rotate + tf_translate + tf_shift_inv))
    img_transform.transform = tf_final

    if tf_final is not None:
        # The function apply_transform only accepts float for cval,
        # so mask the pixels with an unlikely value to exist in an
        # image and apply true multi-channel cval afterwards
        temp_cval = -900.0

        for i in range(0, len(images)):
            # Note: preserve range is important for example for mask images
            order = ImageInterpolation.NEAREST.value if interpolations is None or i > len(interpolations) else interpolations[i].value

            images[i] = skitransform.warp(image=images[i],
                                          inverse_map=tf_final.inverse,
                                          order=order,
                                          mode=fill_mode,
                                          cval=temp_cval,
                                          preserve_range=True).astype(img_dtype)

            # Fix the temporary cvalue to the real cvalue
            if img_data_format == 'channels_first':
                mask = images[i][0, :, :] == temp_cval
            elif img_data_format == 'channels_last':
                mask = images[i][:, :, 0] == temp_cval
            else:
                raise ValueError('Invalid img_data_format: {}'.format(img_data_format))

            images[i][mask] = cvals[i]

    # Check that the image properties are as assumed, same dtype as coming in, values in correct range etc
    for i in range(0, len(images)):
        try:
            np_check_image_properties(images[0], min_val=0.0, max_val=255.0, height=img_height, width=img_width, dtype=img_dtype)
        except ImageValidationError as e:
            if e.error == ImageValidationErrorType.OUT_OF_RANGE_VALUES:
                min_val = np.min(images[i])
                max_val = np.max(images[i])
                print 'WARNING: Found values outside of range [0, 255] after augmentation: [{}, {}] - clipping'.format(min_val, max_val)
                images[i] = np.clip(images[i], 0.0, 255.0)
            else:
                raise e

    return images, img_transform


def np_random_channel_shift(x, intensity, min_c=0.0, max_c=255.0, channel_axis=0):
    # type: (np.ndarray, float, float, float, int) -> np.ndarray

    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_c, max_c) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def np_crop_image(np_img, x1, y1, x2, y2):
    # type: (np.ndarray, int, int, int, int) -> np.ndarray

    """
    Crops an image represented as a Numpy array. The function expects the numpy array
    in dimensions: HxWxC

    # Arguments
        :param np_img: 3 dimensional Numpy array with shape HxWxC
        :param x1: horizontal top left corner of crop
        :param y1: vertical top left corner of crop
        :param x2: horizontal bottom right corner of crop
        :param y2: vertical bottom right corner of crop
    # Returns
        :return: The crop of the image as a Numpy array
    """
    y_size = np_img.shape[0]
    x_size = np_img.shape[1]

    # Sanity check
    if (x1 > x_size or
        x2 > x_size or
        x1 < 0 or
        x2 < 0 or
        y1 > y_size or
        y2 > y_size or
        y1 < 0 or
        y2 < 0):
        raise ValueError('Invalid crop parameters for image shape: {}, ({}, {}), ({}, {})'.format(np_img.shape, x1, y1, x2, y2))

    return np_img[y1:y2, x1:x2]


def np_crop_image_with_fill(np_img, x1, y1, x2, y2, cval):
    # type: (np.ndarray, int, int, int, int, np.ndarray) -> np.ndarray

    """
    Crops an image represented as a Numpy array. Fills the over reaching values with cval.
    The function expects the numpy array in dimensions: HxWxC

    # Arguments
        :param np_img: 3 dimensional Numpy array with shape HxWxC
        :param x1: horizontal top left corner of crop
        :param y1: vertical top left corner of crop
        :param x2: horizontal bottom right corner of crop
        :param y2: vertical bottom right corner of crop
        :param cval: the value to use for filling the pixels that possibly go over
    # Returns
        :return: The crop of the image as a Numpy array
    """

    if x1 >= x2 or y1 >= y2:
        raise ValueError('Invalid crop coordinates; min coordinates bigger or equal to max: {}, {}'.format((y1, x1), (y2, x2)))

    y_size = np_img.shape[0]
    x_size = np_img.shape[1]

    crop_y_size = y2 - y1
    crop_x_size = x2 - x1

    np_cropped_img = np_crop_image(np_img, x1=max(0, x1), y1=max(0, y1), x2=min(x_size, x2), y2=min(y_size, y2))
    v_pad_before = 0 if y1 >= 0 else abs(y1)
    v_pad_after = 0 if y2 <= y_size else y2 - y_size
    h_pad_before = 0 if x1 >= 0 else abs(x1)
    h_pad_after = 0 if x2 <= x_size else x2 - x_size

    if v_pad_before > 0 or v_pad_after > 0 or h_pad_before > 0 or h_pad_after > 0:
        np_cropped_img = np_pad_image(np_img=np_cropped_img,
                                      v_pad_before=v_pad_before,
                                      v_pad_after=v_pad_after,
                                      h_pad_before=h_pad_before,
                                      h_pad_after=h_pad_after,
                                      cval=cval)

    if np_cropped_img.shape[0] != crop_y_size or np_cropped_img.shape[1] != crop_x_size:
        raise ValueError('Cropped and filled image shape does not match with crop size: {}, {}'.format(np_cropped_img.shape, (crop_y_size, crop_x_size)))

    return np_cropped_img


def np_resize_image_with_padding(np_img, shape, cval, interp='bilinear'):
    # type: (np.ndarray, tuple[int], np.ndarray, str) -> np.ndarray

    """
    Scales the image to the desired shape filling the overflowing area with the provided constant
    color value.

    # Arguments
        :param np_img: the image as a numpy array
        :param shape: desired shape
        :param cval: the value to use for filling the pixels that possibly go over due to aspect ratio mismatch
        :param interp: interpolation type ‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’
    # Returns
        :return: The resized image as a numpy array
    """

    if np_img.shape[0] == shape[0] and np_img.shape[1] == shape[1]:
        return np_img

    # Scale so that the bigger dimension matches
    sfactor = float(max(shape[0], shape[1])) / float(max(np_img.shape[0], np_img.shape[1]))

    # If the image's bigger dimension already matches - we only need padding
    if sfactor == 1:
        np_img_resized = np_img
    else:
        target_shape = (int(round(sfactor*np_img.shape[0])), int(round(sfactor*np_img.shape[1])))

        # Do the resizing using PIL because scipy/numpy lacks interpolation
        pil_img = array_to_img(np_img, scale=False)
        func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
        pil_img = pil_img.resize(size=(target_shape[1], target_shape[0]), resample=func[interp])
        np_img_resized = img_to_array(pil_img)

    # Pad to the final desired shape afterwards
    np_img_resized = np_pad_image_to_shape(np_img_resized, shape=shape, cval=cval)

    return np_img_resized


def np_pad_image_to_shape(np_img, shape, cval):
    # type: (np.ndarray, (int,int), np.ndarray) -> np.ndarray

    """
    Pads the image evenly on every side until it matches the dimensions given in
    the shape parameter. If the padding doesn't go evenly the extra is on the left
    side and the bottom.

    # Arguments
        :param np_img: 3 dimensional Numpy array with shape HxWxC
        :param shape: the output shape of the padded image HxW
        :param cval: the color value that is used in the padding
    # Returns
        :return: the padded version of the image
    """
    v_diff = max(0, shape[0] - np_img.shape[0])
    h_diff = max(0, shape[1] - np_img.shape[1])

    v_pad_before = v_diff / 2
    v_pad_after = (v_diff / 2) + (v_diff % 2)

    h_pad_before = h_diff / 2
    h_pad_after = (h_diff / 2) + (h_diff % 2)

    return np_pad_image(np_img, v_pad_before, v_pad_after, h_pad_before, h_pad_after, cval)


def np_pad_image(np_img, v_pad_before, v_pad_after, h_pad_before, h_pad_after, cval):
    # type: (np.array, int, int, int, int, np.array) -> np.array

    """
    Pads the given Numpy array to a given shape and fills the padding with cval
    color value.

    # Arguments:
        :param np_img: 3 dimensional Numpy array with shape HxWxC
        :param v_pad_before: vertical padding on top
        :param v_pad_after: vertical padding on bottom
        :param h_pad_before: horizontal padding on left
        :param h_pad_after: horizontal padding on right
        :param cval: the color value that is used in the padding
    # Returns
        :return: the padded version of the image
    """

    # Temporary value for cval for simplicity
    temp_cval = 919191.0

    np_img = np.pad(np_img, [(v_pad_before, v_pad_after), (h_pad_before, h_pad_after), (0, 0)], 'constant',
                    constant_values=temp_cval)

    # Create a mask for all the temporary cvalues
    cval_mask = np_img[:, :, 0] == temp_cval

    # Replace the temporary cvalues with real color values
    np_img[cval_mask] = cval

    return np_img


def np_from_255_to_normalized(val):
    # type: (np.ndarray) -> np.ndarray

    # From [0,255] to [0,1] to [-0.5,0.5] and then to [-1,1]
    return ((val/255.0) - 0.5) * 2.0


def np_from_normalized_to_255(val):
    # type: (np.ndarray) -> np.ndarray

    # Move to range [0,2] to [0,1] and then to [0,255]
    return ((val+1.0)/2.0) * 255.0


def np_normalize_image_channels(img_array, per_channel_mean=None, per_channel_stddev=None, clamp_to_range=False, inplace=False):
    # type: (np.ndarray, np.ndarray, np.ndarray, bool, bool) -> np.ndarray

    """
    Normalizes the color channels from the given image to zero-centered
    range [-1, 1] from the original [0, 255] range. If the per channels
    mean is provided it is subtracted from the image after zero-centering.
    Furthermore if the per channel standard deviation is given it is
    used to normalize each feature value to a z-score by dividing the given
    data.

    # Arguments
        :param img_array: image to normalize, channels in range [0, 255]
        :param per_channel_mean: per-channel mean of the dataset in range [-1, 1]
        :param per_channel_stddev: per-channel standard deviation in range [-1, 1]
        :param clamp_to_range: should the values be clamped to range [-1, 1]
        :param inplace: should we modify the passed value or create a copy
    # Returns
        :returns: the normalized image with channels in  range [-1, 1]
    """
    if inplace:
        normalized_img_array = img_array.astype(np.float32)
    else:
        normalized_img_array = copy.deepcopy(img_array).astype(np.float32)

    if np.min(normalized_img_array) < 0 or np.max(normalized_img_array) > 255:
        raise ValueError('Image values are not in range [0, 255], got [{}, {}]'.format(np.min(normalized_img_array), np.max(normalized_img_array)))

    normalized_img_array = ((normalized_img_array/255.0) - 0.5) * 2.0

    # Subtract the per-channel-mean from the batch to "center" the data.
    if per_channel_mean is not None:
        _per_channel_mean = np.array(per_channel_mean).astype(np.float32)

        # Per channel mean is in range [-1,1]
        if (_per_channel_mean >= -1.0 - 1e-7).all() and (_per_channel_mean <= 1.0 + 1e-7).all():
            normalized_img_array -= _per_channel_mean
        # Per channel mean is in range [0, 255]
        elif (_per_channel_mean >= 0.0).all() and (_per_channel_mean <= 255.0).all():
            normalized_img_array -= np_from_255_to_normalized(_per_channel_mean)
        else:
            raise ValueError('Per channel mean is in unknown range: {}'.format(_per_channel_mean))

    # Additionally, you ideally would like to divide by the sttdev of
    # that feature or pixel as well if you want to normalize each feature
    # value to a z-score.
    if per_channel_stddev is not None:
        _per_channel_stddev = np.array(per_channel_stddev).astype(np.float32)

        # Per channel stddev is in range [-1, 1]
        if (_per_channel_stddev >= -1.0 - 1e-7).all() and (_per_channel_stddev <= 1.0 + 1e-7).all():
            normalized_img_array /= _per_channel_stddev
        # Per channel stddev is in range [0, 255]
        elif (_per_channel_stddev >= 0.0).all() and (_per_channel_stddev <= 255.0).all():
            normalized_img_array /= np_from_255_to_normalized(_per_channel_stddev)
        else:
            raise ValueError('Per-channel stddev is in unknown range: {}'.format(_per_channel_stddev))

    if clamp_to_range:
        min_val = np.min(normalized_img_array)
        max_val = np.max(normalized_img_array)

        if min_val < -1.0 or max_val > 1.0:
            print 'WARNING: Values outside of range [-1.0, 1.0] were found after normalization - clipping: [{}, {}]'.format(min_val, max_val)
            normalized_img_array = np.clip(normalized_img_array, -1.0, 1.0, out=normalized_img_array)

    # Sanity check for the image values, we shouldn't have any NaN or inf values
    if np.any(np.isnan(normalized_img_array)):
        raise ValueError('NaN values found in image after normalization')

    if np.any(np.isinf(normalized_img_array)):
        raise ValueError('Inf values found in image after normalization')

    return normalized_img_array


def np_get_random_crop_area(np_image, crop_width, crop_height):
    # type: (np.ndarray, int, int) -> ((int, int), (int, int))

    """
    The function returns a random crop from the image as (x1, y1), (x2, y2).

    # Arguments
        :param np_image: image as a numpy array
        :param crop_width: width of the crop
        :param crop_height: height of the crop

    # Returns
        :return: two integer tuples describing the crop: (x1, y1), (x2, y2)
    """

    if crop_width > np_image.shape[1] or crop_height > np_image.shape[0]:
        raise ValueError('Crop dimensions are bigger than image dimensions: [{},{}] vs '.format(crop_height, crop_width, np_image.shape))

    x1 = np.random.randint(0, np_image.shape[1] - crop_width + 1)
    y1 = np.random.randint(0, np_image.shape[0] - crop_height + 1)
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    return (x1, y1), (x2, y2)


def np_get_slic_segmentation(np_img, n_segments, sigma=0.8, compactness=2, max_iter=20, normalize_img=False, borders_only=False):
    # type: (np.ndarray, int, int, float) -> np.ndarray

    """
    Returns the SLIC superpixel segmentation for the parameter image as a numpy array.
    The segmentation is an integer array with dimensions HxW, where each superpixel is
    encoded as a unique integer.

    # Arguments
        :param np_img: the image
        :param n_segments: number of segments to generate
        :param sigma: sigma for SLIC
        :param compactness: compactness for SLIC
        :param max_iter: maximum iterations for the kNN of the SLIC
    # Returns
        :return: the superpixel segmentation
    """

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = slic(normalized_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)
    else:
        segments = slic(np_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)

    if borders_only:
        segments = np.invert(find_boundaries(segments, mode='thick')).astype(np.int32)

    return segments


def np_get_felzenswalb_segmentation(np_img, scale=1, sigma=0.8, min_size=20, multichannel=True, normalize_img=False, borders_only=False):
    # type: (np.ndarray, float, float, int, bool) -> np.ndarray

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = felzenszwalb(image=normalized_img, scale=scale, sigma=sigma, min_size=min_size, multichannel=multichannel)
    else:
        segments = felzenszwalb(image=np_img, scale=scale, sigma=sigma, min_size=min_size, multichannel=multichannel)

    if borders_only:
        segments = np.invert(find_boundaries(segments, mode='thick')).astype(np.int32)

    return segments


def np_get_watershed_segmentation(np_img, markers, compactness=0.001, normalize_img=False, borders_only=False):
    # type: (np.ndarray, int, float) -> np.ndarray

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        gradient = sobel(rgb2gray(normalized_img))
    else:
        gradient = sobel(rgb2gray(np_img))

    segments = watershed(gradient, markers=markers, compactness=compactness)

    if borders_only:
        segments = np.invert(find_boundaries(segments, mode='thick')).astype(np.int32)

    return segments


def np_get_quickshift_segmentation(np_img, kernel_size=3, max_dist=6, sigma=0, ratio=0.5, normalize_img=False, borders_only=False):
    # type: (np.ndarray, float, float, float, float) -> np.ndarray

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = quickshift(normalized_img, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma, ratio=ratio)
    else:
        segments = quickshift(np_img, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma, ratio=ratio)

    if borders_only:
        segments = np.invert(find_boundaries(segments, mode='thick')).astype(np.int32)

    return segments
