# coding=utf-8

import numpy as np
import copy

from skimage.segmentation import slic, felzenszwalb, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel

from keras.preprocessing.image import array_to_img, img_to_array, flip_axis, apply_transform, transform_matrix_offset_center, random_channel_shift

import dataset_utils


def np_apply_random_transform(images,
                              cvals,
                              fill_mode='constant',
                              img_data_format='channels_last',
                              rotation_range=None,
                              zoom_range=None,
                              width_shift_range=0.0,
                              height_shift_range=0.0,
                              channel_shift_ranges=None,
                              horizontal_flip=False,
                              vertical_flip=False):
    # type: (list[np.array], list[np.array], str, str, (float,float), (float,float), float, float, list[np.array], bool, bool) -> list[np.array]

    """
    Randomly augments, in the same way, a list of numpy images.

    # Arguments
        :param images: a list of 3D tensors, image colors in range [0,255]
        :param cvals: the fill values for the images, should be the same size as images list
        :param fill_mode: how to fill the image
        :param img_data_format: format of the image data (channels_last or channels_first)
        :param rotation_range: range of the rotations in degrees
        :param zoom_range: zoom range > 1 zoom in, < 1 zoom out
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
            raise ValueError('Unmatching fill value dimensions for image element {}: {} vs {}'
                             .format(i, len(cvals[i]), images[i].shape[img_channel_axis]))

    # Make sure the images have the same dimensions HxW
    img_width = images[0].shape[img_col_axis]
    img_height = images[0].shape[img_row_axis]

    for i in range(1, len(images)):
        if img_height != images[i].shape[img_row_axis] or img_width != images[i].shape[img_col_axis]:
            raise ValueError('Unmatching image dimensions - cannot apply same transformations: {} vs {}'
                             .format(images[0].shape, images[i].shape))

    # Rotation
    if rotation_range:
        theta = np.pi / 180.0 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0.0

    # Height shift
    if height_shift_range is not None and height_shift_range > 0.0:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * images[0].shape[img_row_axis]
    else:
        tx = 0

    # Width shift
    if width_shift_range is not None and width_shift_range > 0.0:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * images[0].shape[img_col_axis]
    else:
        ty = 0

    # Zoom
    if zoom_range is None or (zoom_range[0] == 1 and zoom_range[1] == 1):
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None

    # Apply rotation to the transformation matrix
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    # Apply translation to the transformation matrix
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

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
            final_transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            images[i] = apply_transform(images[i], final_transform_matrix, img_channel_axis, fill_mode=fill_mode, cval=temp_cval)
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

    # Random channel shifts
    if channel_shift_ranges is not None:
        if len(channel_shift_ranges) > len(images):
            raise ValueError('Channel shift ranges list is longer than the image list: {} vs {}'.format(len(channel_shift_ranges), len(images)))

        for i in range(0, len(channel_shift_ranges)):
            if channel_shift_ranges[i] is None:
                continue

            # Images are [0,255] color encoded, multiply intensity [0,1] by 255 to get the real shift intensity
            images[i] = random_channel_shift(images[i], intensity=channel_shift_ranges[i]*255.0, channel_axis=img_channel_axis)

    # Check that we don't have any NaN values
    for i in range(0, len(images)):
        if np.any(np.invert(np.isfinite(images[i]))):
            raise ValueError('NaN/inifinite values found after applying random transform')

    return images


def np_crop_image(np_img, x1, y1, x2, y2):
    # type: (np.array, int, int, int, int) -> np.array

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


def np_scale_image_with_padding(np_img, shape, cval, interp='bilinear'):
    # type: (np.array, tuple[int], np.array, str) -> np.array

    """
    Scales the image to the desired shape filling the overflowing area with the provided constant
    color value.

    # Arguments
        :param np_img:
        :param shape: Desired shape
        :param cval: The value to use for filling the pixels that possibly go over due to aspect ratio mismatch
        :param interp: interpolation type ‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’
    # Returns
        :return: The resized image as a numpy array
    """

    # Scale so that the bigger dimension matches
    sfactor = float(max(shape[0], shape[1])) / float(max(np_img.shape[0], np_img.shape[1]))

    # If the image's bigger dimension already matches
    if sfactor == 1:
        np_img_resized = np_img
    else:
        target_shape = (int(round(sfactor*np_img.shape[0])), int(round(sfactor*np_img.shape[1])))

        # Do the resizing using PIL because scipy/numpy lacks interpolation
        pil_img = array_to_img(np_img)
        func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
        pil_img = pil_img.resize(size=(target_shape[1], target_shape[0]), resample=func[interp])
        np_img_resized = img_to_array(pil_img)

    # Pad to the final desired shape afterwards
    np_img_resized = np_pad_image_to_shape(np_img_resized, shape=shape, cval=cval)

    return np_img_resized


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


def np_normalize_image_channels(img_array, per_channel_mean=None, per_channel_stddev=None, clamp_to_range=False, inplace=False):
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
        normalized_img_array = img_array
    else:
        normalized_img_array = copy.deepcopy(img_array)

    # Subtract the per-channel-mean from the batch to "center" the data.
    if per_channel_mean is not None:
        _per_channel_mean = np.array(per_channel_mean)
        if not ((_per_channel_mean < 1.0 + 1e-7).all() and (_per_channel_mean > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel mean is not within range [-1, 1]')
        normalized_img_array -= dataset_utils.np_from_normalized_to_255(_per_channel_mean)

    # Additionally, you ideally would like to divide by the sttdev of
    # that feature or pixel as well if you want to normalize each feature
    # value to a z-score.
    if per_channel_stddev is not None:
        _per_channel_stddev = np.array(per_channel_stddev)
        if not ((_per_channel_stddev < 1.0 + 1e-7).all() and (_per_channel_stddev > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel stddev is not within range [-1, 1]')
        normalized_img_array /= dataset_utils.np_from_normalized_to_255(_per_channel_stddev)

    normalized_img_array -= 128.0
    normalized_img_array /= 128.0

    if clamp_to_range:
        np.clip(normalized_img_array, -1.0, 1.0, out=normalized_img_array)

    # Sanity check for the image values, we shouldn't have any NaN or inf values
    if np.any(np.isnan(normalized_img_array)):
        raise ValueError('NaN values found in image after normalization')

    if np.any(np.isinf(normalized_img_array)):
        raise ValueError('Inf values found in image after normalization')

    return normalized_img_array


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


def np_get_slic_segmentation(np_img, n_segments, sigma=0.8, compactness=2, max_iter=20, normalize_img=False):
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

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = slic(normalized_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)
    else:
        segments = slic(np_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)

    return segments


def np_get_felzenswalb_segmentation(np_img, scale=1, sigma=0.8, min_size=20, multichannel=True, normalize_img=False):
    # type: (np.array, float, float, int, bool) -> np.array

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = felzenszwalb(image=normalized_img, scale=scale, sigma=sigma, min_size=min_size, multichannel=multichannel)
    else:
        segments = felzenszwalb(image=np_img, scale=scale, sigma=sigma, min_size=min_size, multichannel=multichannel)

    return segments


def np_get_watershed_segmentation(np_img, markers, compactness=0.001, normalize_img=False):
    # type: (np.array, int, float) -> np.array

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        gradient = sobel(rgb2gray(normalized_img))
    else:
        gradient = sobel(rgb2gray(np_img))

    segments = watershed(gradient, markers=markers, compactness=compactness)

    return segments


def np_get_quickshift_segmentation(np_img, kernel_size=3, max_dist=6, sigma=0, ratio=0.5, normalize_img=False):
    # type: (np.array, float, float, float, float) -> np.array

    if normalize_img:
        normalized_img = np_normalize_image_channels(np_img, clamp_to_range=True)
        segments = quickshift(normalized_img, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma, ratio=ratio)
    else:
        segments = quickshift(np_img, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma, ratio=ratio)

    return segments
