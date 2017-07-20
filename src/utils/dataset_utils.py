# coding=utf-8

import os
import random
import multiprocessing
import math

import numpy as np

from PIL import Image
from joblib import Parallel, delayed
from keras.preprocessing.image import load_img, img_to_array

import image_utils

##############################################
# UTILITY CLASSES
##############################################


class MaterialClassInformation(object):
    def __init__(self, material_id, substance_ids, substance_names, r_color_values, color):
        self.id = material_id

        self.substance_ids = substance_ids
        self.substance_names = substance_names
        self.r_color_values = r_color_values

        self.name = substance_names[0]
        self.color = color


##############################################
# UTILITY FUNCTIONS
##############################################


def load_material_class_information(material_labels_file_path):
    # type: (str) -> list[MaterialClassInformation]

    """
    Loads the material class information from a file. The information should be in the format:

        substance_id,substance_name,red_color;substance_id,substance_name,red_color ...
        substance_id,substance_name,red_color;substance_id,substance_name,red_color ...

    The semicolon ';' is used to combine multiple material colors under a single category. In case
    materials are combined under one category the name of the material is the name of the first
    material listed. The material ID will be the zero-based index of the material within the file.

    # Arguments
        :param material_labels_file_path: path to the material information file (CSV)
    # Returns
        :return: a list of MaterialClassInformation objects
    """

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
        r_color_values = [int(x) for x in material_params[2]]

        # The id is the index of the material in the file, this index will determine
        # the dimension index in the mask image for this material class
        material_id = i - 1

        materials.append(
            MaterialClassInformation(
                material_id,
                tuple(substance_ids),
                tuple(substance_names),
                tuple(r_color_values),
                get_color_for_category_index(material_id)))

    return materials


def get_number_of_parallel_jobs(max=32):
    """
    Returns the optimal number of parallel jobs by inspecting the available CPU
    count.

    # Arguments
        :param max: maximum number of jobs
    # Returns
        :return: number of jobs that scales well for the platform
    """
    num_cores = multiprocessing.cpu_count()
    n_jobs = min(max, num_cores)
    return n_jobs


def get_bit(n, i):
    # type: (int, int) -> bool

    """
    Returns a boolen indicating whether the nth bit in the integer i was on or not.

    # Arguments
        :param n: index of the bit
        :param i: integer
    # Returns
        :return: true if the nth bit was on false otherwise.
    """
    return n & (1 << i) > 0


def get_color_for_category_index(idx):
    # type: (int) -> np.array[np.uint8]

    """
    Returns a color for class index. The index is used in the red channel
    directly and the green and blue channels are derived similar to the
    PASCAL VOC coloring algorithm.

    # Arguments
        :param idx: Index of the class
    # Returns
        :return: RGB color in channel range [0,255] per channel
    """

    cid = idx
    r = idx
    g = 0
    b = 0

    for j in range(0, 7):
        g = g | get_bit(cid, 0) << 7 - j
        b = b | get_bit(cid, 1) << 7 - j
        cid = cid >> 2

    color = np.array([r, g, b], dtype=np.uint8)
    return color


def get_files(path, ignore_hidden_files=True):
    # type: (str, bool) -> list[str]

    """
    Returns a list of all the files in the directory. Subdirectories are excluded.

    # Arguments
        :param path: path to the directory
        :param ignore_hidden_files: should hidden files starting with '.' be ignored.
    # Returns
        :return: a list of files in the directory
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Filter hidden files such as .DS_Store
    if ignore_hidden_files:
        files = [f for f in files if not f.startswith('.')]
    return files


def calculate_per_channel_mean(file_paths, num_channels):
    # type: (list[str], int) -> np.array[np.float32]

    """
    Calculates the per-channel mean from all the images in the given
    path and returns it as a num_channels dimensional numpy array.

    # Arguments
        :param file_paths: paths to the files to be used in the calculation
        :param num_channels: number of channels in the image 1,3 or 4
    # Returns
        :return: numpy array with the channel means in range [-1, 1]
    """

    # Parallelize per-channel sum calculations
    n_jobs = get_number_of_parallel_jobs()

    data = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_calculate_per_channel_mean)(
            fp, num_channels) for fp in file_paths)

    # Calculate the final value
    sums = np.sum(np.vstack(data), axis=0)
    color_tot = sums[:-1]
    px_tot = sums[-1]

    per_channel_mean = color_tot / px_tot
    print 'Per-channel mean calculation complete: {}'.format(per_channel_mean)

    return per_channel_mean


def _calculate_per_channel_mean(path, num_channels):
    img = load_n_channel_image(path, num_channels)
    img_array = img_to_array(img)

    # Normalize color channels to zero-centered range [-1, 1]
    img_array = image_utils.np_normalize_image_channels(img_array)

    # Accumulate the sums of the different color channels
    tot = np.array([0.0] * (num_channels + 1))

    # Store the color value sums
    for i in range(0, num_channels):
        tot[i] = np.sum(img_array[:, :, i])

    # In the final index store the number of pixels
    tot[num_channels] = img_array.shape[0] * img_array.shape[1]

    return tot


def calculate_per_channel_stddev(file_paths, per_channel_mean, num_channels):
    # type: (list[str], np.array[np.float32], int) -> np.array[np.float32]

    """
    Calculates the per-channel standard deviation from all the images in the given
    path and returns it as a num_channels dimensional numpy array.

    # Arguments
        :param file_paths: paths to the files to be used in the calculation
        :param num_channels: number of channels in the image 1,3 or 4
    # Returns
        :return: numpy array with the channel means in range [-1, 1]
    """

    # Parallelize per-channel variance calculations
    n_jobs = get_number_of_parallel_jobs()

    data = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_calculate_per_channel_stddev)(
            fp, per_channel_mean, num_channels) for fp in file_paths)

    # Calculate the final value
    sums = np.sum(np.vstack(data), axis=0)
    var_tot = sums[:-1]
    px_tot = sums[-1]

    # Calculate final variance value
    per_channel_var = var_tot / px_tot
    print 'Final per-channel variance: {}'.format(per_channel_var)

    # Calculate the stddev
    per_channel_stddev = np.sqrt(per_channel_var)
    print 'Per-channel stddev calculation complete: {}'.format(per_channel_stddev)

    return per_channel_stddev


def _calculate_per_channel_stddev(path, per_channel_mean, num_channels):
    var_tot = np.array([0.0] * (num_channels + 1))

    # Load the image as numpy array
    img = load_n_channel_image(path, num_channels)
    img_array = img_to_array(img)

    # Normalize colors to zero-centered range [-1, 1]
    img_array = image_utils.np_normalize_image_channels(img_array)

    # Var: SUM_0..N {(val-mean)^2} / N
    for i in range(0, num_channels):
        var_tot[i] = np.sum(np.square(img_array[:, :, i] - per_channel_mean[i]))

    # Accumulate the number of total pixels
    var_tot[-1] = img_array.shape[0] * img_array.shape[1]

    return var_tot


def calculate_median_frequency_balancing_weights(file_paths, material_class_information):
    # type: (str, list[str], list[MaterialClassInformation]) -> np.array[np.float32]

    """
    Calculates the median frequency balancing weights for provided
    segmentation masks.

    # Arguments
        :param file_paths: paths to the mask files
        :param files: segmentation masks
        :param material_class_information: material class information of the dataset
    # Returns
        :return: a Numpy array with N_CLASSES values each describing the weight for that specific class
    """

    n_jobs = get_number_of_parallel_jobs()

    data = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_calculate_mask_class_frequencies)(
            fp, material_class_information) for fp in file_paths)
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


def _calculate_mask_class_frequencies(mask_file_path, material_class_information):
    img_array = img_to_array(load_img(mask_file_path))
    expanded_mask = expand_mask(img_array, material_class_information)
    class_pixels = np.sum(expanded_mask, axis=(0, 1))

    # Select all classes which appear in the picture i.e.
    # have a value over zero
    num_pixels = img_array.shape[0] * img_array.shape[1]
    img_pixels = (class_pixels > 0.0) * num_pixels

    return class_pixels, img_pixels


def expand_mask(np_mask_img, material_class_information, verbose=False):
    # type: (np.array, list[MaterialClassInformation], bool) -> np.array[np.float32]

    """
    Expands the segmentation color mask from the color image to a numpy array of
    HxWxNUM_CLASSES where each channel contains a one hot encoded layer of that class.
    The class information is stored to the channel index of the corresponding material id.

    # Arguments
        :param np_mask_img: segmentation mask as a Numpy image RGB with channels in range [0, 255]
        :param material_class_information: an array with the material class information
        :param verbose: should the function print information about the run
    # Returns
        :return: expanded one-hot encoded mask where each mask is on it's own layer
    """

    num_material_classes = len(material_class_information)
    expanded_mask = np.zeros(shape=(np_mask_img.shape[0], np_mask_img.shape[1], num_material_classes), dtype=np.float32)
    found_materials = [] if verbose else None

    # Go through each material class
    for material_class in material_class_information:

        # Initialize a color mask with all false
        class_mask = np.zeros(shape=(np_mask_img.shape[0], np_mask_img.shape[1]), dtype='bool')

        # Go through each possible color for that class and create a mask
        # of the pixels that contain a color of the possible values.
        # Note: many colors are possible because some classes maybe collapsed
        # together to form a single class
        for r_color_val in material_class.r_color_values:
            # The substance/material category information is in the red
            # color channel in the opensurfaces dataset
            class_mask |= np_mask_img[:, :, 0] == r_color_val

        # Set the activations of all the pixels that match the color mask to 1
        # on the dimension that matches the material class id
        if np.any(class_mask):
            if found_materials is not None:
                found_materials.append(material_class.substance_ids)
            expanded_mask[:, :, material_class.id][class_mask] = 1.0

    if verbose:
        print 'Found {} materials with the following substance ids: {}\n'.format(len(found_materials), found_materials)

    return expanded_mask


def index_encode_mask(np_mask_img, material_class_information, verbose=False):
    # type: (np.array, list[MaterialClassInformation], bool) -> np.array[np.int32]

    """
    Expands the segmentation color mask from the color image to a numpy array of
    HxWxNUM_CLASSES where each channel contains a one hot encoded layer of that class.
    The class information is stored to the channel index of the corresponding material id.

    # Arguments
        :param np_mask_img: segmentation mask as a Numpy image RGB with channels in range [0, 255]
        :param material_class_information: an array with the material class information
        :param verbose: should the function print information about the run
    # Returns
        :return:
    """
    index_mask = np.zeros(shape=(np_mask_img.shape[0], np_mask_img.shape[1]), dtype=np.int32)
    found_materials = [] if verbose else None

    # Go through each material class
    for material_class in material_class_information:

        # Initialize a color mask with all false
        class_mask = np.zeros(shape=(np_mask_img.shape[0], np_mask_img.shape[1]), dtype='bool')

        # Go through each possible color for that class and create a mask
        # of the pixels that contain a color of the possible values.
        # Note: many colors are possible because some classes maybe collapsed
        # together to form a single class
        for r_color_val in material_class.r_color_values:
            # The substance/material category information is in the red
            # color channel in the opensurfaces dataset
            class_mask |= np_mask_img[:, :, 0] == r_color_val

        # Set the activations of all the pixels that match the color mask to 1
        # on the dimension that matches the material class id
        if np.any(class_mask):
            if found_materials is not None:
                found_materials.append(material_class.substance_ids)
            index_mask[class_mask] = material_class.id

    if verbose:
        print 'Found {} materials with the following substance ids: {}\n'.format(len(found_materials), found_materials)

    return index_mask


def get_number_of_batches(data_set_size, batch_size):
    # type: (int, int) -> int

    """
    Returns the number of batches for the given dataset size and batch size.
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


def get_batch_index_range(data_set_size, batch_size, batch_index):
    # type: (int, int, int) -> (int, int)

    """
    Returns the start and end indices to the data for the given dataset size,
    batch size and batch index. The function assumes that all data will be used
    every epoch and the last batch size can be smaller than the others.

    # Arguments
        :param data_set_size: size of the data set
        :param batch_size: size of a single batch
        :param batch_index: index of the current batch
    # Returns
        :return: a tuple of integers describing the (start, end) indices to the dataset
    """

    start = batch_index * batch_size
    end = min(data_set_size, (batch_index + 1) * batch_size)
    return start, end


def get_batch_from_data_set(data_set, batch_size, batch_index, loop=False):
    # type: (list, int, int, bool) -> list

    """
    Returns the start and end indices to the data for the given data set size,
    batch size and batch index. The function assumes that all data will be used every
    epoch and, if looping is not True, that the last batch size can be smaller
    than the others.

    # Arguments
        :param data_set: the data set as a list of entries e.g. pairs (x,y)
        :param batch_size: size of a single batch
        :param batch_index: index of the current batch
        :param loop: whether to loop to the beginning of the data if index goes over.
    # Returns
        :return: a slice of the list with the elements in the batch
    """

    data_set_size = len(data_set)
    start = batch_index * batch_size
    end = min(data_set_size, (batch_index + 1) * batch_size)

    if loop and end - start < batch_size:
        return data_set[start:end] + data_set[:batch_size - (end-start)]

    return data_set[start:end]


def split_labeled_dataset(photo_files, mask_files, split):
    # type: (list[str], list[str], list[float]) -> (list[str], list[str], list[str])

    """
    Splits the whole dataset randomly into three different groups: training,
    validation and test, according to the split provided as the parameter.
    The provided photo and segmentation lists should have matching filenames so
    that after sorting the arrays have matching pairs in matching indices.

    # Arguments
        :param photo_files: photo files
        :param mask_files: segmentation mask files
        :param split: a list of floats describing the dataset split, must sum to one: [training, validation, test]
    # Returns
        :return: a tuple of filename arrays describing the sets: (training, validation, test)
    """

    if len(photo_files) != len(mask_files):
        raise ValueError('Unmatching photo - mask file list sizes: photos: {}, masks: {}'
                         .format(len(photo_files), len(mask_files)))

    if sum(split) != 1.0:
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
    if training_set_size + validation_set_size + test_set_size != dataset_size:
        diff = dataset_size - (training_set_size + validation_set_size + test_set_size)
        training_set_size += diff

    if training_set_size + validation_set_size + test_set_size != dataset_size:
        raise ValueError('The split set sizes do not sum to total dataset size: {} + {} + {} = {} != {}'
                         .format(training_set_size,
                                 validation_set_size,
                                 test_set_size,
                                 training_set_size + validation_set_size + test_set_size,
                                 dataset_size))

    training_set = photo_mask_files[0:training_set_size]
    validation_set = photo_mask_files[training_set_size:training_set_size + validation_set_size]
    test_set = photo_mask_files[training_set_size + validation_set_size:]

    return training_set, validation_set, test_set


def load_n_channel_image(path, num_channels):
    # type: (str, int) -> PIL.Image

    """
    Returns a PIL image with 1,3 or 4 image channels.

    # Arguments
        :param path: path to the image file
        :param num_channels: number of channels
    # Returns
        :return: a PIL image with num channels
    """

    if not (num_channels == 4 or num_channels == 3 or num_channels == 1):
        raise ValueError('Number of channels must be 1, 3 or 4')

    if num_channels == 3:
        return load_img(path)

    return Image.open(path)


def get_required_image_dimensions(current_image_shape, div2_constraint):
    # type: ((int, int), int) -> (int, int)

    """
    Returns the closest (bigger) required image dimensions that satisfy
    the given divisibility by two constraint i.e. is n times divisible by
    two.

    # Arguments
        :param current_image_shape: current image shape (HxW)
        :param div2_constraint: divisible by two constraint
    # Returns
        :return: the required image dimensions to satisfy the div2 constraint (HxW)
    """

    padded_height = get_closest_higher_number_with_n_trailing_zeroes(current_image_shape[0], div2_constraint)
    padded_width = get_closest_higher_number_with_n_trailing_zeroes(current_image_shape[1], div2_constraint)
    padded_shape = (padded_height, padded_width)

    return padded_shape


def get_closest_higher_number_with_n_trailing_zeroes(num, n):
    # type: (int, int) -> int

    """
    Returns the closest (higher) number with n trailing zeroes in the binary.
    This can be used to ensure continuous divisibility by two n times.

    # Arguments
        :param num: the number
        :param n: how many trailing zeroes are required
    # Returns
        :return: the closest higher number that satisfies the n trailing zeroes constraint
    """

    if count_trailing_zeroes(num) >= n:
        return num

    smallest_num_with_n_zeros = 1 << n

    if num < smallest_num_with_n_zeros:
        return smallest_num_with_n_zeros

    remainder = num % smallest_num_with_n_zeros
    return num + (smallest_num_with_n_zeros - remainder)


def count_trailing_zeroes(num):
    # type: (int) -> int

    """
    Counts the trailing zeroes in the binary of the paramter number.

    # Arguments
        :param num: the number
    # Returns
        :return: the number of trailing zeroes
    """

    zeroes = 0
    while ((num >> zeroes) & 1) == 0:
        zeroes = zeroes + 1
    return zeroes


def np_from_255_to_normalized(val):
    # type: (np.array) -> np.array

    # From [0,255] to [-128,128] and then to [-1,1]
    val -= 128.0
    val /= 128.0
    return val


def np_from_normalized_to_255(val):
    # type: (np.array) -> np.array

    # Move to range [0,1] and then to [0,255]
    val = (val+1.0)/2.0
    val *= 255.0
    return val


def normalize_batch(batch, per_channel_mean=None, per_channel_stddev=None, clamp_to_range=False):
    # type: (np.array, np.array, np.array) -> np.array

    """
    Standardizes the color channels from the given image batch to zero-centered
    range [-1, 1] from the original [0, 255] range. In case a parameter is not supplied
    that normalization is not applied.

    # Arguments
        :param batch: numpy array with a batch of images to normalize
        :param per_channel_mean: per channel mean in range [-1,1]
        :param per_channel_stddev: per channel standard deviation in range [-1,1]
        :param clamp_to_range: should the values be clamped to range [-1,1]
    # Returns
        :return: The parameter batch normalized with the given values
    """

    # TODO: Refactor to take values in range [0,255]
    if per_channel_mean is not None:
        if not ((per_channel_mean < 1.0 + 1e-7).all() and (per_channel_mean > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel mean is not within range [-1, 1]')
        batch -= np_from_normalized_to_255(per_channel_mean)

    if per_channel_stddev is not None:
        if not ((per_channel_stddev < 1.0 + 1e-7).all() and (per_channel_stddev > -1.0 - 1e-7).all()):
            raise ValueError('Per-channel stddev is not within range [-1, 1]')
        batch /= np_from_normalized_to_255(per_channel_stddev + 1e-7)

    batch -= 128.0
    batch /= 128.0

    if clamp_to_range:
        np.clip(batch, -1.0, 1.0, out=batch)

    return batch
