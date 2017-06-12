import os
import random
import multiprocessing
import threading

import numpy as np

from PIL import Image
from joblib import Parallel, delayed
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import flip_axis, apply_transform, transform_matrix_offset_center
from keras import backend as K

##############################################
# UTILITY CLASSES
##############################################

"""
Takes an iterator/generator and makes it thread-safe by
serializing call to the `next` method of given iterator/generator.
"""


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


"""
A decorator that takes a generator function and makes it thread-safe.
"""


def threadsafe_flow(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def unwrap_get_segmentation_data_pair(arg, **kwarg):
    return SegmentationDataGenerator.get_segmentation_data_pair(*arg, **kwarg)


class MaterialClassInformation(object):
    def __init__(
            self,
            material_id,
            substance_ids,
            substance_names,
            color_values):
        self.id = material_id

        self.substance_ids = substance_ids
        self.substance_names = substance_names
        self.color_values = color_values

        self.name = substance_names[0]


class SegmentationDataGenerator(object):
    def __init__(
            self,
            photo_files_folder_path,
            mask_files_folder_path,
            photo_mask_files,
            material_class_information,
            random_seed=None,
            per_channel_mean_normalization=True,
            per_channel_mean=None,
            per_channel_stddev_normalization=True,
            per_channel_stddev=None,
            use_data_augmentation=False,
            augmentation_probability=0.5,
            rotation_range=0.,
            zoom_range=0.,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='constant',
            photo_cval=None,
            mask_cval=None):

        self.img_data_format = K.image_data_format()

        if self.img_data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3

        if self.img_data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.photo_files_folder_path = photo_files_folder_path
        self.mask_files_folder_path = mask_files_folder_path
        self.photo_mask_files = photo_mask_files
        self.material_class_information = material_class_information

        self.per_channel_mean_normalization = per_channel_mean_normalization

        # Ensure the per_channel_mean is a numpy tensor
        if (per_channel_mean is not None):
            self.per_channel_mean = np.array(per_channel_mean)

        self.per_channel_stddev_normalization = per_channel_stddev_normalization

        # Ensure per_channel_stddev is a numpy tensor
        if (per_channel_stddev is not None):
            self.per_channel_stddev = np.array(per_channel_stddev)

        self.use_data_augmentation = use_data_augmentation
        self.augmentation_probability = augmentation_probability
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.fill_mode = fill_mode
        self.photo_cval = photo_cval
        self.mask_cval = mask_cval

        # Calculate missing per-channel mean if necessary
        if (per_channel_mean_normalization and per_channel_mean is None):
            photos = [sample[0] for sample in photo_mask_files]
            self.per_channel_mean = calculate_per_channel_mean(photo_files_folder_path, photos)

        # Calculate missing per-channel stddev if necessary
        if (per_channel_stddev_normalization and per_channel_stddev is None):
            photos = [sample[0] for sample in photo_mask_files]
            self.per_channel_stddev = calculate_per_channel_stddev(photo_files_folder_path, photos)

        # Use per-channel mean but in range [0, 255] if nothing else is given.
        # The normalization is done to the whole batch after transformations so
        # the images are not in range [-1,1] before transformations.
        if (self.photo_cval is None):
            self.photo_cval = ((np.array(self.per_channel_mean) + 1.0) / 2.0) * 255.0

        # Use black (background)
        if (self.mask_cval is None):
            self.mask_cval = (0.0, 0.0, 0.0)

        # Use the given random seed for reproducibility
        if (random_seed is not None):
            np.random.seed(random_seed)

        # Zoom range can either be a tuple or a scalar
        if np.isscalar(zoom_range):
            self.zoom_range = [1.0 - zoom_range, 1.0 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    """
    Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single photo image.
            y: 3D tensor, a single mask image.
        # Returns
            Inputs (x, y) with the same random transform applied.
    """

    def apply_random_transform(self, x, y):

        # x and y are a single images, so they don't have the batch dimension
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # Rotation
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        # Zoom
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        # Apply rotation to the transformation matrix
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        # Apply zoom to the tranformation matrix
        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        # Apply the tranformation matrix to the image
        if transform_matrix is not None:
            # The function apply_transform only accepts float for cval,
            # so mask the pixels with an unlikely value to exist in an
            # image and apply true multi-channel cval afterwards
            temp_cval = 919191.0

            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=temp_cval)
            mask = x[:, :, 0] == temp_cval
            x[mask] = self.photo_cval

            y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=temp_cval)
            mask = y[:, :, 0] == temp_cval
            y[mask] = self.mask_cval

        # Apply at random a horizontal flip to the image
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                y = flip_axis(y, img_col_axis)

        # Apply at random a vertical flip to the image
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                y = flip_axis(y, img_row_axis)

        # Check that we don't have any nan values
        if (np.any(np.isnan(x)) or np.any(np.isnan(y))):
            raise ValueError('NaN values found after applying random transform')

        return x, y

    """
    Standardizes a photo batch
    """

    def standardize_batch(self, batch):
        # Standardizes the color channels from the given image to zero-centered
        # range [-1, 1] from the original [0, 255] range.
        batch -= 128
        batch /= 128

        # Subtract the per-channel-mean from the batch
        # to "center" the data.
        if self.per_channel_mean_normalization:
            if self.per_channel_mean is not None:
                batch -= self.per_channel_mean
            else:
                raise ValueError(
                    'SegmentationDataGenerator specifies `per_channel_mean_normalization` but has not been fit on any training data.')

        # Additionally, you ideally would like to divide by the stddev of
        # that feature or pixel as well if you want to normalize each feature
        # value to a z-score.
        if self.per_channel_stddev_normalization:
            if self.per_channel_stddev is not None:
                batch /= (self.per_channel_stddev + 1e-7)
            else:
                raise ValueError(
                    'SegmentationDataGenerator specifies `per_channel_stddev_normalization` but has not been fit on any training data.')

        return batch

    def get_segmentation_data_pair(self, photo_mask_pair, crop_size, div2_constraint=4):

        # Load the image and mask as PIL images
        image = load_img(os.path.join(self.photo_files_folder_path, photo_mask_pair[0]))
        mask = load_img(os.path.join(self.mask_files_folder_path, photo_mask_pair[1]))

        # Resize the image to match the mask size if necessary, since
        # the original photos are sometimes huge
        if (image.size != mask.size):
            orig_size = image.size
            image = image.resize(mask.size, Image.ANTIALIAS)

        if (image.size != mask.size):
            raise ValueError('Non-matching image and mask dimensions after resize: {} vs {}'
                             .format(image.size, mask.size))

        # Convert to numpy array
        image = img_to_array(image)
        mask = img_to_array(mask)

        # If we are using data augmentation apply the random tranformation
        # to both the image and mask now. We apply the transformation to the
        # whole image to decrease the number of 'dead' pixels due to tranformations
        # within the possible crop.
        if (self.use_data_augmentation and
                    np.random.random() <= self.augmentation_probability):
            image, mask = self.apply_random_transform(image, mask)
            # Save augmented images for debug
            # array_to_img(image).save('aug_{}'.format(photo_mask_pair[0]))
            # array_to_img(mask).save('aug_{}'.format(photo_mask_pair[1]))

        # If a crop size is given: Take a random crop
        # of both the image and the mask
        if crop_size is not None:

            if (count_trailing_zeroes(crop_size[0]) < div2_constraint or
                        count_trailing_zeroes(crop_size[1]) < div2_constraint):
                raise ValueError('The crop size does not satisfy the div2 constraint of {}'.format(div2_constraint))

            try:
                # Re-attempt crops if the crops end up getting only background
                # i.e. black pixels
                attempts = 5

                for i in range(0, attempts):
                    x1 = np.random.randint(0, image.shape[1] - crop_size[0])
                    y1 = np.random.randint(0, image.shape[0] - crop_size[1])
                    x2 = x1 + crop_size[0]
                    y2 = y1 + crop_size[1]

                    mask_crop = np_crop_image(mask, x1, y1, x2, y2)

                    # If the mask crop is only background (all R channel is zero) - try another crop
                    if (np.max(mask_crop[:, :, 0]) == 0 and i < attempts - 1):
                        continue

                    mask = mask_crop
                    image = np_crop_image(image, x1, y1, x2, y2)
                    break

            except IOError:
                raise IOError(
                    'Could not load image or mask from pair: {}, {}'.format(photo_mask_pair[0], photo_mask_pair[1]))

        # Save crops for debug
        # array_to_img(image).save('crop_{}'.format(photo_mask_pair[0]))
        # array_to_img(mask).save('crop_{}'.format(photo_mask_pair[1]))

        # If a crop size is not given, make sure the image dimensions satisfy
        # the div2_constraint i.e. are n times divisible by 2 to work within
        # the network. If the dimensions are not ok pad the images.
        if (count_trailing_zeroes(image.shape[0]) < div2_constraint or
            count_trailing_zeroes(image.shape[1]) < div2_constraint):

            padded_height = get_closest_number_with_n_trailing_zeroes(image.shape[0], div2_constraint)
            padded_width = get_closest_number_with_n_trailing_zeroes(image.shape[1], div2_constraint)

            v_diff = padded_height - image.shape[0]
            h_diff = padded_width - image.shape[1]

            v_pad_before = v_diff / 2
            v_pad_after = (v_diff / 2) + (v_diff % 2)

            h_pad_before = h_diff / 2
            h_pad_after = (h_diff / 2) + (h_diff % 2)

            # Mask should be padded with the mask_cval color if
            # available if not, use black
            mask_cval = self.mask_cval if (self.mask_cval is not None) else (0, 0, 0)
            mask = np_pad_image(mask, v_pad_before, v_pad_after, h_pad_before, h_pad_after, mask_cval)

            # Image needs to be filled with the photo_cval color if
            # available if not, use black
            img_cval = self.photo_cval if (self.photo_cval is not None) else (0, 0, 0)
            image = np_pad_image(image, v_pad_before, v_pad_after, h_pad_before, h_pad_after, img_cval)

        # Expand the mask image to accommodate different classes
        # H x W x NUM_CLASSES
        mask = expand_mask(mask, self.material_class_information)

        return image, mask

    @threadsafe_flow
    def get_flow(self, batch_size, crop_size=None):
        # Calculate the number of batches that we can create from this data
        num_batches = len(self.photo_mask_files) // batch_size

        num_cores = multiprocessing.cpu_count()
        n_jobs = min(32, num_cores)

        while True:
            # Shuffle the (photo, mask) -pairs after every epoch
            random.shuffle(self.photo_mask_files)

            for i in range(0, num_batches):
                # The files for this batch
                batch_files = self.photo_mask_files[i * batch_size:(i + 1) * batch_size]

                # Parallel processing of the files in this batch
                data = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(unwrap_get_segmentation_data_pair)((self, pair, crop_size)) for pair in batch_files)

                # Note: all the examples in the batch have to have the same dimensions
                X, Y = zip(*data)
                X, Y = self.standardize_batch(np.array(X)), np.array(Y)

                yield X, Y


##############################################
# UTILITY FUNCTIONS
##############################################

"""
Returns all the files (filenames) found in the path.
Does not include subdirectories.
"""


def get_files(path, ignore_hidden_files=True):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Filter hidden files such as .DS_Store
    if ignore_hidden_files:
        files = [f for f in files if not f.startswith('.')]
    return files


def get_closest_number_with_n_trailing_zeroes(num, n):
    if (count_trailing_zeroes(num) >= n):
        return num

    smallest_num_with_n_zeros = 1 << n

    if (num < smallest_num_with_n_zeros):
        return smallest_num_with_n_zeros

    remainder = num % smallest_num_with_n_zeros
    return num + (smallest_num_with_n_zeros - remainder)


def count_trailing_zeroes(num):
    zeroes = 0
    while (((num >> zeroes) & 1) == 0):
        zeroes = zeroes + 1
    return zeroes


"""
Crops an image represented as a Numpy array. Note that when handling images
as a Numpy arrays the dimensions are HxWxC
"""


def np_crop_image(np_img, x1, y1, x2, y2):
    y_size = np_img.shape[0]
    x_size = np_img.shape[1]

    # Sanity check
    if (x1 >= x_size or
                x2 >= x_size or
                x1 < 0 or
                x2 < 0 or
                y1 >= y_size or
                y2 >= y_size or
                y1 < 0 or
                y2 < 0):
        raise ValueError('Invalid crop parameters for image shape: {}, ({}, {}, {}, {}'
                         .format(np_img.shape, x1, y1, x2, y2))

    return np_img[y1:y2, x1:x2]


def np_pad_image(np_img, v_pad_before, v_pad_after, h_pad_before, h_pad_after, cval):
    # Temporary value for cval for simplicity
    temp_cval = 919191.0

    np_img = np.pad(
        np_img,
        [(v_pad_before, v_pad_after), (h_pad_before, h_pad_after), (0, 0)],
        'constant',
        constant_values=temp_cval)

    # Create a mask for all the temporary cvalues
    cval_mask = np_img[:, :, 0] == temp_cval

    # Replace the temporary cvalues with real color values
    np_img[cval_mask] = cval

    return np_img


"""
Normalizes the color channels from the given image to zero-centered
range [-1, 1] from the original [0, 255] range. If the per channels
mean is provided it is subtracted from the image after zero-centering.
Furthermore if the per channel standard deviation is given it is
used to normalize each feature value to a z-score by dividing the given
data.
    # Arguments
        TODO
    # Returns
        TODO
"""


def normalize_image_channels(img_array, per_channel_mean=None, per_channel_stddev=None):
    img_array -= 128
    img_array /= 128

    if (per_channel_mean != None):
        # Subtract the per-channel-mean from the batch
        # to "center" the data.
        img_array -= per_channel_mean

    if (per_channel_stddev != None):
        # Additionally, you ideally would like to divide by the sttdev of
        # that feature or pixel as well if you want to normalize each feature
        # value to a z-score.
        img_array /= (per_channel_stddev + 1e-7)

    # Sanity check for the image values, we shouldn't have any NaN or inf values
    if (np.any(np.isnan(img_array))):
        raise ValueError('NaN values found in image after normalization')

    if (np.any(np.isinf(img_array))):
        raise ValueError('Inf values found in image after normalization')

    return img_array


"""
Calculates the per-channel mean from all the images in the given
path and returns it as a 3 dimensional numpy array.

Parameters to the function:

path - the path to the image files
files - the files to be used in the calculation
"""


def calculate_per_channel_mean(path, files):
    # Continue from saved data if there is
    px_tot = 0.0
    color_tot = np.array([0.0, 0.0, 0.0])

    for idx in range(0, len(files)):
        f = files[idx]

        if idx % 10 == 0 and idx != 0:
            print 'Processed {} images: px_tot: {}, color_tot: {}'.format(idx, px_tot, color_tot)
            print 'Current per-channel mean: {}'.format(color_tot / px_tot)

        # Load the image as numpy array
        img = load_img(os.path.join(path, f))
        img_array = img_to_array(img)

        # Normalize colors to zero-centered range [-1, 1]
        img_array = normalize_image_channels(img_array)

        # Accumulate the number of total pixels
        px_tot += img_array.shape[0] * img_array.shape[1]

        # Accumulate the sums of the different color channels
        color_tot[0] += np.sum(img_array[:, :, 0])
        color_tot[1] += np.sum(img_array[:, :, 1])
        color_tot[2] += np.sum(img_array[:, :, 2])

    # Calculate the final value
    per_channel_mean = color_tot / px_tot
    print 'Per-channel mean calculation complete: {}'.format(per_channel_mean)

    return per_channel_mean


'''
Calculates the per-channel-stddev
'''


def calculate_per_channel_stddev(path, files, per_channel_mean):
    # Calculate variance
    px_tot = 0.0
    var_tot = np.array([0.0, 0.0, 0.0])

    for idx in range(0, len(files)):
        f = files[idx]

        if idx % 10 == 0 and idx != 0:
            print 'Processed {} images: px_tot: {}, var_tot: {}\n'.format(idx, px_tot, var_tot)
            print 'Current per channel variance: {}\n'.format(var_tot / px_tot)

        # Load the image as numpy array
        img = load_img(os.path.join(path, f))
        img_array = img_to_array(img)

        # Normalize colors to zero-centered range [-1, 1]
        img_array = normalize_image_channels(img_array)

        # Accumulate the number of total pixels
        px_tot += img_array.shape[0] * img_array.shape[1]

        # Var: SUM_0..N {(val-mean)^2} / N
        var_tot[0] += np.sum(np.square(img_array[:, :, 0] - per_channel_mean[0]))
        var_tot[1] += np.sum(np.square(img_array[:, :, 1] - per_channel_mean[1]))
        var_tot[2] += np.sum(np.square(img_array[:, :, 2] - per_channel_mean[2]))

    # Calculate final variance value
    per_channel_var = var_tot / px_tot
    print 'Final per-channel variance: {}'.format(per_channel_var)

    # Calculate the stddev
    per_channel_stddev = np.sqrt(per_channel_var)
    print 'Per-channel stddev calculation complete: {}'.format(per_channel_stddev)

    return per_channel_stddev


def calculate_mask_class_frequencies(mask_file_path, material_class_information):
    img_array = img_to_array(load_img(mask_file_path))
    expanded_mask = expand_mask(img_array, material_class_information)
    class_pixels = np.sum(expanded_mask, axis=(0, 1))

    # Select all classes which appear in the picture i.e.
    # have a value over zero
    num_pixels = img_array.shape[0] * img_array.shape[1]
    img_pixels = (class_pixels > 0.0) * num_pixels

    return (class_pixels, img_pixels)


"""
Calculates the median frequency balancing weights
"""


def calculate_median_frequency_balancing_weights(path, files, material_class_information):
    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    data = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(calculate_mask_class_frequencies)(
            os.path.join(path, f),
            material_class_information) for f in files)
    data = zip(*data)

    class_pixels = data[0]
    img_pixels = data[1]
    class_pixels = np.sum(class_pixels, axis=0)
    img_pixels = np.sum(img_pixels, axis=0)

    # freq(c) is the number of pixels of class c divided
    # by the total number of pixels in images where c is present.
    # Median freq is the median of these frequencies.
    class_frequencies = class_pixels / img_pixels
    median_frequency = np.median(class_frequencies)
    median_frequency_weights = median_frequency / class_frequencies

    return median_frequency_weights


def load_material_class_information(material_labels_file_path):
    materials = []

    with open(material_labels_file_path, 'r') as f:
        content = f.readlines()

    # Remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # First line is header so start from 1
    for i in range(1, len(content)):
        # Each line is of form
        #
        # substance_id,substance_name,red_color;substance_id,substance_name,red_color
        # The semicolon ';' is used to combine multiple material colors under a
        # single category. In case materials are combined under one category
        subcategories = content[i].split(';')
        material_params = [f.split(',') for f in subcategories]
        material_params = zip(*material_params)

        substance_ids = [int(x) for x in material_params[0]]
        substance_names = [x for x in material_params[1]]
        color_values = [int(x) for x in material_params[2]]

        # The id is the index of the material in the file, this index will determine
        # the dimension index in the mask image for this material class
        materials.append(
            MaterialClassInformation(i - 1, tuple(substance_ids), tuple(substance_names), tuple(color_values)))

    return materials


"""
The material information in the mask is encoded into the red color channel.
Parameters to the function:

mask - a numpy array of the segmentation mask image WxHx3
material_class_information - an array which has the relevent MaterialClassInformation objects

The functions returns an object of size:

MASK_HEIGHT x MASK_WIDTH x NUM_MATERIAL CLASSES
"""


def expand_mask(mask, material_class_information, verbose=False):
    num_material_classes = len(material_class_information)
    expanded_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], num_material_classes), dtype='float32')
    found_materials = [] if verbose else None

    # Go through each material class
    for material_class in material_class_information:

        # Initialize a color mask with all false
        class_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype='bool')

        # Go through each possible color for that class and create a mask
        # of the pixels that contain a color of the possible values.
        # Note: many colors are possible because some classes maybe collapsed
        # together to form a single class
        for color in material_class.color_values:
            # The substance/material category information is in the red
            # color channel in the opensurfaces dataset
            class_mask |= mask[:, :, 0] == color

        # Set the activations of all the pixels that match the color mask to 1
        # on the dimension that matches the material class id
        if (np.any(class_mask)):
            if (found_materials != None):
                found_materials.append(material_class.substance_ids)
            expanded_mask[:, :, material_class.id][class_mask] = 1.0

    if (verbose):
        print 'Found {} materials with the following substance ids: {}\n'.format(len(found_materials), found_materials)

    return expanded_mask


def flatten_mask(expanded_mask, material_class_information, verbose=False):
    # The predictions now reflect material class ids
    predictions = np.argmax(expanded_mask, axis=-1)

    # TODO: Generalize to N channel images
    flattened_mask = np.zeros(shape=(predictions.shape[0], predictions.shape[1], 3), dtype='uint8')
    num_found_materials = 0

    for material_class in material_class_information:
        material_class_id = material_class.id

        # Select all the pixels with the corresponding id values
        class_mask = predictions[:, :] == material_class_id

        # Set all the corresponding pixels in the flattened image
        # to the material color. If there are many colors for one
        # material, select the first to represent them all.
        material_r_color = material_class.color_values[0]

        # Parse a unique color for the material
        color = None

        if material_class_id == 0:
            color = (material_r_color, 0, 0)
        else:
            color = (material_r_color, np.random.randint(10, 256), np.random.randint(10, 256))

        if (verbose and np.any(class_mask)):
            print 'Found material: {}, assigning it color: {}'.format(material_class.name, color)
            num_found_materials += 1

        # Assign the material color to all the masked pixels
        flattened_mask[class_mask] = color

    if (verbose):
        print 'Found in total {} materials'.format(num_found_materials)

    return flattened_mask


"""
Splits the whole dataset randomly into three different groups: training,
validation and test, according to the split provided as the parameter.

Returns three lists of photo - mask pairs:

0 training
1 validation
2 test
"""


def split_dataset(
        photo_files,
        mask_files,
        split):
    if (len(photo_files) != len(mask_files)):
        raise ValueError(
            'Unmatching photo - mask file list sizes: photos: {}, masks: {}'.format(len(photo_files), len(mask_files)))

    if (sum(split) != 1.0):
        raise ValueError('The given dataset split does not sum to 1: {}'.format(sum(split)))

    # Sort the files by name so we have matching photo - mask files
    photo_files.sort()
    mask_files.sort()

    # Zip the lists to create a list of matching photo - mask file tuples
    photo_mask_files = zip(photo_files, mask_files)

    # Shuffle the list of files
    random.shuffle(photo_mask_files)

    # Divide the dataset to three different parts: training, validation and test
    # according to the given split: 0=training, 1=validation, 2=test
    dataset_size = len(photo_mask_files)
    training_set_size = int(round(split[0] * dataset_size))
    validation_set_size = int(round(split[1] * dataset_size))
    test_set_size = int(round(split[2] * dataset_size))

    # If the sizes don't match exactly add/subtract the different
    # from the training set
    if (training_set_size + validation_set_size + test_set_size != dataset_size):
        diff = dataset_size - (training_set_size + validation_set_size + test_set_size)
        training_set_size += diff

    if (training_set_size + validation_set_size + test_set_size != dataset_size):
        raise ValueError(
            'The split set sizes do not sum to total dataset size: {} + {} + {} = {} != {}'.format(training_set_size,
                                                                                                   validation_set_size,
                                                                                                   test_set_size,
                                                                                                   training_set_size + validation_set_size + test_set_size,
                                                                                                   dataset_size))

    training_set = photo_mask_files[0:training_set_size]
    validation_set = photo_mask_files[training_set_size:training_set_size + validation_set_size]
    test_set = photo_mask_files[training_set_size + validation_set_size:]

    return training_set, validation_set, test_set
