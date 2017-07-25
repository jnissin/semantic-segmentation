# coding=utf-8

import numpy as np
from skimage.segmentation import slic

from keras.preprocessing.image import flip_axis, apply_transform, transform_matrix_offset_center


def np_apply_random_transform(images,
                              cvals,
                              fill_mode='constant',
                              img_data_format='channels_last',
                              rotation_range=None,
                              zoom_range=None,
                              horizontal_flip=False,
                              vertical_flip=False):
    # type: (list[np.array], list[np.array], str, str, (float,float), (float,float), bool, bool) -> list[np.array]

    """
    Randomly augments, in the same way, a list of numpy images.

    # Arguments
        :param images: a list of 3D tensors, image colors in range [0,255]
        :param cvals: the fill values for the images, should be the same size as images list
        :param fill_mode: how to fill the image
        :param img_data_format: format of the image data (channels_last or channels_first)
        :param rotation_range: range of the rotations in degrees
        :param zoom_range: zoom range > 1 zoom in, < 1 zoom out
        :param horizontal_flip: should horizontal flips be applied
        :param vertical_flip: should vertical flips be applied
    # Returns
        :return: Inputs (x, y) with the same random transform applied.
    """

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
            raise ValueError('Unmatching fill value dimensions for image element {}: {} vs {}'
                             .format(i, len(cvals[i]), images[i].shape[img_channel_axis]))

    # Rotation
    if rotation_range:
        theta = np.pi / 180.0 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0.0

    # Zoom
    if zoom_range is None or (zoom_range[0] == 1 and zoom_range[1] == 1):
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    # Apply rotation to the transformation matrix
    transform_matrix = None

    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    # Apply zoom to the transformation matrix
    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    # Apply the transformation matrix to the image
    if transform_matrix is not None:
        # The function apply_transform only accepts float for cval,
        # so mask the pixels with an unlikely value to exist in an
        # image and apply true multi-channel cval afterwards
        temp_cval = 919191.0

        for i in range(0, len(images)):
            h, w = images[i].shape[img_row_axis], images[i].shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            images[i] = apply_transform(images[i], transform_matrix, img_channel_axis,
                                        fill_mode=fill_mode, cval=temp_cval)
            mask = images[i][:, :, 0] == temp_cval
            images[i][mask] = cvals[i]

    # Apply at random a horizontal flip to the image
    if horizontal_flip:
        if np.random.random() < 0.5:
            for i in range(0, len(images)):
                images[i] = flip_axis(images[i], img_col_axis)

    # Apply at random a vertical flip to the image
    if vertical_flip:
        if np.random.random() < 0.5:
            for i in range(0, len(images)):
                images[i] = flip_axis(images[i], img_row_axis)

    # Check that we don't have any nan values
    for i in range(0, len(images)):
        if np.any(np.invert(np.isfinite(images[i]))):
            raise ValueError('NaN/inifinite values found after applying random transform')

    return images


def np_crop_image(np_img, x1, y1, x2, y2):
    # type: (np.array, int, int, int, int) -> np
    # .array

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
        raise ValueError('Invalid crop parameters for image shape: {}, ({}, {}, {}, {}'
                         .format(np_img.shape, x1, y1, x2, y2))

    return np_img[y1:y2, x1:x2]


def np_pad_image_to_shape(np_img, shape, cval):
    # type: (np.array, (int,int), np.array) -> np.array

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


def np_normalize_image_channels(img_array, per_channel_mean=None, per_channel_stddev=None, clamp_to_range=False):
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
    # Returns
        :returns: the normalized image with channels in  range [-1, 1]
    """
    img_array -= 128.0
    img_array /= 128.0

    # Subtract the per-channel-mean from the batch to "center" the data.
    if per_channel_mean is not None:
        if not ((per_channel_mean < 1.0 + 1e-7).all() and (per_channel_mean > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel mean is not within range [-1, 1]')
        img_array -= per_channel_mean

    # Additionally, you ideally would like to divide by the sttdev of
    # that feature or pixel as well if you want to normalize each feature
    # value to a z-score.
    if per_channel_stddev is not None:
        if not ((per_channel_stddev < 1.0 + 1e-7).all() and (per_channel_stddev > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel stddev is not within range [-1, 1]')
        img_array /= (per_channel_stddev + 1e-7)

    if clamp_to_range:
        np.clip(img_array, -1.0, 1.0, out=img_array)

    # Sanity check for the image values, we shouldn't have any NaN or inf values
    if np.any(np.isnan(img_array)):
        raise ValueError('NaN values found in image after normalization')

    if np.any(np.isinf(img_array)):
        raise ValueError('Inf values found in image after normalization')

    return img_array


def np_get_random_crop_area(np_image, crop_width, crop_height):
    # type: (np.array, int, int) -> ((int, int), (int, int))

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


def np_get_superpixel_segmentation(np_img, n_segments, sigma=5, compactness=10.0, max_iter=10):
    # type: (np.array, int, int, float) -> np.array

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

    # Apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(np_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)
    return segments
