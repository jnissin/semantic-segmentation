# coding=utf-8

import os
import random
import multiprocessing
import time

import numpy as np

from PIL import Image
from joblib import Parallel, delayed

from image_utils import load_img, img_to_array, ImageTransform

from .. import settings
from ..data_set import ImageFile
from ..enums import CoordinateType, ClassWeightType

##############################################
# GLOBALS
##############################################

_per_channel_mean_processed = multiprocessing.Value('i', 0)
_per_channel_stddev_processed = multiprocessing.Value('i', 0)
_material_samples_processed = multiprocessing.Value('i', 0)


##############################################
# UTILITY CLASSES
##############################################

class BoundingBox(object):

    def __init__(self, y_min, x_min, y_max, x_max, coordinate_type):

        if y_min >= y_max:
            raise ValueError('Invalid Y values: y_min >= y_max: {} >= {}'.format(y_min, y_max))

        if x_min >= x_max:
            raise ValueError('Invalid X values: x_min >= x_max: {} >= {}'.format(x_min, x_max))

        self._y_min = y_min
        self._x_min = x_min
        self._y_max = y_max
        self._x_max = x_max
        self._coordinate_type = coordinate_type

    @property
    def y_min(self):
        return self._y_min

    @property
    def x_min(self):
        return self._x_min

    @property
    def y_max(self):
        return self._y_max

    @property
    def x_max(self):
        return self._x_max

    @property
    def top_left(self):
        return self._y_min, self._x_min

    @property
    def top_right(self):
        return self._y_min, self._x_max

    @property
    def bottom_right(self):
        return self._y_max, self._x_max

    @property
    def bottom_left(self):
        return self._y_max, self._x_min

    @property
    def corners(self):
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left

    @property
    def coordinate_type(self):
        # type: () -> CoordinateType
        return self._coordinate_type

    def transform(self, transform, min_transformed_bbox_size=4):
        """
        Transforms the bounding box using the parameter ImageTransform. Returns a new
        bounding box describing the transformed bounding box or None if the bounding box
        is smaller than the min_transformed_bbox_size or out of bounds after transform.

        # Arguments
            :param transform: an ImageTransform used to transform this bounding box
            :param min_transformed_bbox_size: minimum size of the transformed bounding box
        # Returns
            :return: a new transformed bounding box
        """

        # type: ImageTransform -> BoundingBox

        # Bounding box as np ndarray: tlc, trc, brc, blc
        np_bbox = np.array((self.top_left, self.top_right, self.bottom_right, self.bottom_left), dtype=np.float32)

        # The transform needs to get the coordinates in (x,y) instead of (y,x), so flip the coordinates there and back
        if self.coordinate_type is CoordinateType.NORMALIZED:
            np_transformed_bbox = np.fliplr(transform.transform_normalized_coordinates(np.fliplr(np_bbox)))
        elif self.coordinate_type is CoordinateType.ABSOLUTE:
            np_transformed_bbox = np.fliplr(transform.transform_coordinates(np.fliplr(np_bbox)))

            # Switch to absolute coordinates
            np_transformed_bbox[:, 0] *= transform.image_height
            np_transformed_bbox[:, 1] *= transform.image_width
            np_transformed_bbox = np.round(np_transformed_bbox).astype(dtype=np.int32)
        else:
            raise ValueError('Invalid coordinate type: {}'.format(self.coordinate_type))

        y_limit = 1.0 if self.coordinate_type is CoordinateType.NORMALIZED else transform.image_height
        x_limit = 1.0 if self.coordinate_type is CoordinateType.ABSOLUTE else transform.image_width

        # Clamp the values of the corners into valid ranges [[0, img_height], [0, img_width]]
        tf_y_min = np.clip(np.min(np_transformed_bbox[:, 0]), 0, y_limit)
        tf_y_max = np.clip(np.max(np_transformed_bbox[:, 0]), 0, y_limit)
        tf_x_min = np.clip(np.min(np_transformed_bbox[:, 1]), 0, x_limit)
        tf_x_max = np.clip(np.max(np_transformed_bbox[:, 1]), 0, x_limit)

        # If the area is less than the minimum transformed bbox size return None
        tf_size = (tf_y_max - tf_y_min) * (tf_x_max - tf_x_min)

        if tf_size <= min_transformed_bbox_size:
            return None

        return BoundingBox(tf_y_min, tf_x_min, tf_y_max, tf_x_max, self.coordinate_type)


class MaterialSample(object):

    def __init__(self,
                 file_name,
                 material_id,
                 material_r_color,
                 image_width,
                 image_height,
                 pixel_info_list):
        # type: (str, int, int, int, int, list[tuple[int]]) -> None

        self.file_name = file_name
        self.material_id = int(material_id)
        self.material_r_color = int(material_r_color)
        self.num_material_pixels = len(pixel_info_list)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        # Note: pixel info list values are tuples of (pixel value, y coord, x coord)
        y_min = int(min(pixel_info_list, key=lambda t: t[1])[1])
        x_min = int(min(pixel_info_list, key=lambda t: t[2])[2])
        y_max = int(max(pixel_info_list, key=lambda t: t[1])[1])
        x_max = int(max(pixel_info_list, key=lambda t: t[2])[2])

        self.yx_min = (y_min, x_min)
        self.yx_max = (y_max, x_max)

        center_y = int((self.yx_min[0] + self.yx_max[0]) / 2)
        center_x = int((self.yx_min[1] + self.yx_max[1]) / 2)
        self.bbox_center = (center_y, center_x)

        bbox_height = int(self.yx_max[0] - self.yx_min[0])
        bbox_width = int(self.yx_max[1] - self.yx_min[1])
        self.bbox_size = bbox_height * bbox_width

    @property
    def file_name_no_ext(self):
        return os.path.splitext(self.file_name)[0]

    @property
    def bbox_top_left_corner_abs(self):
        return self.yx_min

    @property
    def bbox_top_right_corner_abs(self):
        return self.yx_min[0], self.yx_max[1]

    @property
    def bbox_bottom_right_corner_abs(self):
        return self.yx_max

    @property
    def bbox_bottom_left_corner_abs(self):
        return self.yx_max[0], self.yx_min[1]

    @property
    def bbox_top_left_corner_rel(self):
        return float(self.yx_min[0])/float(self.image_height), float(self.yx_min[1])/float(self.image_width)

    @property
    def bbox_top_right_corner_rel(self):
        return float(self.yx_min[0])/float(self.image_height), float(self.yx_max[1])/float(self.image_width)

    @property
    def bbox_bottom_right_corner_rel(self):
        return float(self.yx_max[0])/float(self.image_height), float(self.yx_max[1])/float(self.image_width)

    @property
    def bbox_bottom_left_corner_rel(self):
        return float(self.yx_max[0])/float(self.image_height), float(self.yx_min[1])/float(self.image_width)

    def get_bbox_abs(self):
        # type: () -> BoundingBox
        return BoundingBox(y_min=self.yx_min[0], x_min=self.yx_min[1], y_max=self.yx_max[0], x_max=self.yx_max[1], coordinate_type=CoordinateType.ABSOLUTE)

    def get_bbox_rel(self):
        # type: () -> BoundingBox
        rel_yx_min = self.bbox_top_left_corner_rel
        rel_yx_max = self.bbox_bottom_right_corner_rel
        return BoundingBox(y_min=rel_yx_min[0], x_min=rel_yx_min[1], y_max=rel_yx_max[0], x_max=rel_yx_max[1], coordinate_type=CoordinateType.NORMALIZED)


class MINCSample(object):

    def __init__(self, minc_label, photo_id, x, y):
        # type: (int, str, float, float) -> None

        self.minc_label = int(minc_label)
        self.photo_id = photo_id
        self.x = float(x)
        self.y = float(y)

    @property
    def file_name(self):
        return self.photo_id + '.jpg'


class MaterialClassInformation(object):

    def __init__(self,
                 material_id,
                 substance_ids,
                 substance_names,
                 r_color_values,
                 color):
        self.id = material_id

        self.substance_ids = substance_ids
        self.substance_names = substance_names
        self.r_color_values = r_color_values

        self.name = substance_names[0]
        self.color = color


class SegmentationSetInformation(object):

    def __init__(self,
                 name,
                 labeled_photos,
                 labeled_masks,
                 material_samples=None):
        # type: (str, list[str], list[str], list[list[MaterialSample]]) -> None

        if len(labeled_photos) != len(labeled_masks):
            raise ValueError('Unmatching photo and mask lengths in the data set: {} vs {}'.format(len(labeled_photos), len(labeled_masks)))

        self.name = name
        self.labeled_photos = labeled_photos
        self.labeled_masks = labeled_masks
        self.labeled_size = len(labeled_photos)
        self.material_samples = material_samples
        self.statistics_loaded = False

        # Per-file statistics
        self.total_pixels = 0
        self.class_pixel_frequencies = []

        # Material samples statistics
        self.material_samples_total_pixels = 0
        self.material_samples_class_pixel_frequencies = []
        self.material_samples_class_instance_frequencies = []

    def load_statistics(self, masks_folder_path, red_color_to_material_id, parallelization_backend='threading'):
        # type: (str, dict, str) -> None

        # Find the maximum class id value
        num_classes = max(red_color_to_material_id.values()) + 1

        # Calculate per-file statistics
        n_jobs = get_number_of_parallel_jobs()

        statistics = Parallel(n_jobs=n_jobs, backend=parallelization_backend)(
            delayed(get_mask_statistics)(
                os.path.join(masks_folder_path, mask_file), red_color_to_material_id, num_classes) for mask_file in self.labeled_masks)

        num_pixels, material_frequencies = zip(*statistics)

        self.total_pixels = int(sum(num_pixels))
        self.class_pixel_frequencies = [int(x) for x in sum(material_frequencies)]   # Make JSON encoding compatible

        # Calculate material samples statistics
        if self.material_samples is not None:
            self.material_samples_total_pixels = 0
            self.material_samples_class_pixel_frequencies = [0] * num_classes
            self.material_samples_class_instance_frequencies = [0] * num_classes

            for material_id in range(0, len(self.material_samples)):

                self.material_samples_class_instance_frequencies[material_id] = len(self.material_samples[material_id])

                for material_sample in self.material_samples[material_id]:
                    self.material_samples_total_pixels += material_sample.bbox_size
                    self.material_samples_class_pixel_frequencies[material_id] += material_sample.num_material_pixels

        self.statistics_loaded = True


class SegmentationTrainingSetInformation(SegmentationSetInformation):

    def __init__(self,
                 name,
                 labeled_photos,
                 labeled_masks,
                 unlabeled_photos=[],
                 material_samples=None):
        # type: (str, list[str], list[str], list[str], list[list[MaterialSample]]) -> ()

        super(SegmentationTrainingSetInformation, self).__init__(name, labeled_photos, labeled_masks, material_samples)
        self.unlabeled_photos = unlabeled_photos
        self.unlabeled_size = len(unlabeled_photos)


class SegmentationDataSetInformation(object):

    def __init__(self,
                 training_set,
                 validation_set,
                 test_set,
                 random_seed):
        # type: (SegmentationTrainingSetInformation, SegmentationSetInformation, SegmentationSetInformation, int) -> None

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.random_seed = random_seed

        self.per_channel_mean = None
        self.per_channel_stddev = None
        self.labeled_per_channel_mean = None
        self.labeled_per_channel_stddev = None

    def load_statistics(self, labeled_photos_folder_path, unlabeled_photos_folder_path, masks_folder_path, material_class_information, parallelization_backend='threading', verbose=False):
        """
        Loads the statistics for all the data sets and calculates per-channel mean and stddev for
        labeled only and labeled + unlabeled training set configurations.

        # Arguments

        # Returns
            :return: Nothing
        """

        red_color_to_material_id = get_red_color_to_material_id_lookup_dict(material_class_information)

        if not self.training_set.statistics_loaded:
            self._print('Starting training set statistics calculation', verbose)
            s_time = time.time()
            self.training_set.load_statistics(masks_folder_path=masks_folder_path, red_color_to_material_id=red_color_to_material_id, parallelization_backend=parallelization_backend)
            self._print('Training set statistics calculation finished in: {} s'.format(time.time()-s_time), verbose)

        if not self.validation_set.statistics_loaded:
            self._print('Starting validation set statistics calculation', verbose)
            s_time = time.time()
            self.validation_set.load_statistics(masks_folder_path=masks_folder_path, red_color_to_material_id=red_color_to_material_id, parallelization_backend=parallelization_backend)
            self._print('Validation set statistics calculation finished in: {} s'.format(time.time()-s_time), verbose)

        if not self.test_set.statistics_loaded:
            self._print('Starting test set statistics calculation', verbose)
            s_time = time.time()
            self.test_set.load_statistics(masks_folder_path=masks_folder_path, red_color_to_material_id=red_color_to_material_id, parallelization_backend=parallelization_backend)
            self._print('Test set statistics calculation finished in: {} s'.format(time.time()-s_time), verbose)

        # Calculate per-channel mean for labeled only and full training set
        photo_file_paths = [os.path.join(labeled_photos_folder_path, f) for f in self.training_set.labeled_photos]

        self._print('Starting labeled only per-channel mean calculation for {} training set photo files'.format(len(photo_file_paths)), verbose)
        s_time = time.time()
        per_channel_mean = calculate_per_channel_mean(image_files=photo_file_paths, verbose=verbose, parallelization_backend=parallelization_backend)
        self._print('Labeled only per-channel mean: {} calculation completed in {} s'.format(list(per_channel_mean), time.time()-s_time), verbose)

        self._print('Starting labeled only per-channel stddev calculation for {} training set photo files'.format(len(photo_file_paths)), verbose)
        s_time = time.time()
        per_channel_stddev = calculate_per_channel_stddev(image_files=photo_file_paths, per_channel_mean=per_channel_mean, verbose=verbose, parallelization_backend=parallelization_backend)
        self._print('Labeled only per-channel stddev: {} calculation completed in {} s'.format(list(per_channel_stddev), time.time()-s_time), verbose)

        self.labeled_per_channel_mean = [float(x) for x in per_channel_mean]        # Make JSON encoding compatible
        self.labeled_per_channel_stddev = [float(x) for x in per_channel_stddev]    # Make JSON encoding compatible

        # Calculate per-channel stddev for labeled only and full training set
        if self.training_set.unlabeled_photos is not None and len(self.training_set.unlabeled_photos) > 0:
            unlabeled_photo_file_paths = [os.path.join(unlabeled_photos_folder_path, f) for f in self.training_set.unlabeled_photos]
            photo_file_paths = photo_file_paths + unlabeled_photo_file_paths

            self._print('Starting per-channel mean calculation for {} training set photo files'.format(len(photo_file_paths)), verbose)
            s_time = time.time()
            per_channel_mean = calculate_per_channel_mean(image_files=photo_file_paths, verbose=verbose, parallelization_backend=parallelization_backend)
            self._print('Per-channel mean: {} calculation completed in {} s'.format(list(per_channel_mean), time.time()-s_time), verbose)

            self._print('Starting labeled only per-channel stddev calculation for {} training set photo files'.format(len(photo_file_paths)), verbose)
            s_time = time.time()
            per_channel_stddev = calculate_per_channel_stddev(image_files=photo_file_paths, per_channel_mean=per_channel_mean, verbose=verbose, parallelization_backend=parallelization_backend)
            self._print('Per-channel stddev: {} calculation completed in {} s'.format(list(per_channel_stddev), time.time()-s_time), verbose)

            self.per_channel_mean = [float(x) for x in per_channel_mean]            # Make JSON encoding compatible
            self.per_channel_stddev = [float(x) for x in per_channel_stddev]        # Make JSON encoding compatible

    def get_class_weights(self, class_weight_type, ignore_classes, use_material_samples):
        # type: (ClassWeightType, list, bool) -> list

        ignore_classes = ignore_classes if ignore_classes is not None else []

        if class_weight_type is ClassWeightType.NONE:
            class_weights = list(np.ones(24, dtype=np.float32))
            for idx in ignore_classes:
                class_weights[idx] = 0.0
            return class_weights

        if use_material_samples:
            class_probabilities = self._calculate_material_samples_class_probabilities(ignore_classes=ignore_classes)
        else:
            class_probabilities = self._calculate_class_probabilities(ignore_classes=ignore_classes)

        if class_weight_type is ClassWeightType.MEDIAN_FREQUENCY_BALANCING:
            return self._calculate_mfb_class_weights(class_probabilities=class_probabilities)
        elif class_weight_type is ClassWeightType.ENET and use_material_samples:
            return self._calculate_enet_class_weights(class_probabilities=class_probabilities)
        else:
            raise ValueError('Unknown class weight type: {}'.format(class_weight_type))

    def _calculate_mfb_class_weights(self, class_probabilities):
        class_probabilities = np.array(class_probabilities, dtype=np.float64)
        non_zero_class_probabilities = np.array([p for p in class_probabilities if p > 0.0])
        median_probability = np.median(non_zero_class_probabilities)

        # Calculate class weights and avoid division by zero for ignored classes such as
        # the background
        class_weights = []

        for p in class_probabilities:
            if p > 0.0:
                class_weights.append(median_probability / p)
            else:
                class_weights.append(0.0)

        return class_weights

    def _calculate_enet_class_weights(self, class_probabilities):
        c = 1.02
        class_probabilities = np.array(class_probabilities)

        # Calculate class weights and avoid division by zero for ignored classes such as
        # the background
        class_weights = []

        for p in class_probabilities:
            if p > 0.0:
                class_weights.append(1.0 / np.log(c + p))
            else:
                class_weights.append(0.0)

        return class_weights

    def _calculate_class_probabilities(self, ignore_classes):
        # type: (list) -> list

        total_pixels = self.training_set.total_pixels
        class_pixels = np.array(self.training_set.class_pixel_frequencies, dtype=np.float64)

        for class_id in ignore_classes:
            total_pixels -= class_pixels[class_id]
            class_pixels[class_id] = 0

        class_probabilities = class_pixels / total_pixels
        return list(class_probabilities)

    def _calculate_material_samples_class_probabilities(self, ignore_classes):
        # type: (list) -> list

        if self.training_set.material_samples is None:
            raise ValueError('Cannot calculate material sample class probabilities, training set does not contain material samples')

        total_pixels = self.training_set.material_samples_total_pixels
        class_pixels = np.array(self.training_set.material_samples_class_pixel_frequencies, dtype=np.float64)

        for class_id in ignore_classes:
            total_pixels -= class_pixels[class_id]
            class_pixels[class_id] = 0

        class_probabilities = class_pixels / total_pixels
        return list(class_probabilities)

    def _print(self, s, verbose):
        if verbose:
            print s

##############################################
# UTILITY FUNCTIONS
##############################################

def load_segmentation_data_set_information(data_set_information_file_path):
    # type: (str) -> SegmentationDataSetInformation
    import jsonpickle

    with open(data_set_information_file_path, 'r') as f:
        json_str = f.read()
        data_set_information = jsonpickle.decode(json_str)

        if not isinstance(data_set_information, SegmentationDataSetInformation):
            raise ValueError('Read JSON file could not be decoded as SegmentationDataSetInformation')

    return data_set_information


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


def get_red_color_to_material_id_lookup_dict(material_class_information):
    # type: (list[MaterialClassInformation]) -> dict

    # Create a look up table for material red color -> material id
    r_color_to_material_id = dict()

    for mci in material_class_information:
        for r_color in mci.r_color_values:
            r_color_to_material_id[r_color] = mci.id

    return r_color_to_material_id


def get_number_of_parallel_jobs(override_max=None):
    """
    Returns the optimal number of parallel jobs by inspecting the available CPU
    count.

    # Arguments
        :param override_max: maximum number of jobs which overrides the number in settings, None if settings max used
    # Returns
        :return: number of jobs that scales well for the platform
    """
    num_cores = multiprocessing.cpu_count()

    if override_max is None:
        n_jobs = min(num_cores, settings.MAX_NUMBER_OF_JOBS)
    else:
        n_jobs = min(num_cores, override_max)

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
    # type: (int) -> np.ndarray[np.uint8]

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


def calculate_per_channel_mean(image_files, verbose=False, parallelization_backend='threading'):
    # type: (list[ImageFile], bool, str) -> np.ndarray

    """
    Calculates the per-channel mean from all the given ImageFiles or image paths and returns it as
    a N_CHANNELS dimensional numpy array.

    # Arguments
        :param image_files: ImageFiles to be used in the calculation
        :param verbose: should we print progress information
        :param parallelization_backend: which parallelization method to use 'threading' or 'multiprocessing'
    # Returns
        :return: numpy array with the channel means in range [-1, 1]
    """

    # Parallelize per-channel sum calculations
    n_jobs = get_number_of_parallel_jobs()

    channel_sums = Parallel(n_jobs=n_jobs, backend=parallelization_backend)(
        delayed(_get_per_channel_sum)(image_file, np.uint64, verbose) for image_file in image_files)

    # Calculate the final value
    channel_sums = sum(channel_sums)
    num_pixels = channel_sums[-1]
    channel_sums = channel_sums[:-1]
    per_channel_mean = channel_sums.astype(dtype=np.float128) / float(num_pixels)
    per_channel_mean = per_channel_mean.astype(dtype=np.float32)

    # Normalize to range [-1.0, 1.0]
    per_channel_mean = per_channel_mean / 255.0
    per_channel_mean = per_channel_mean - 0.5
    per_channel_mean = per_channel_mean * 2.0

    if verbose:
        global _per_channel_mean_processed
        _per_channel_mean_processed.value = 0

    return per_channel_mean


def _get_per_channel_sum(image_file, dtype=np.uint64, verbose=False):
    # type: (ImageFile, np.dtype, bool) -> np.ndarray

    if isinstance(image_file, ImageFile):
        img = image_file.get_image()
    elif isinstance(image_file, str):
        img = load_img(image_file)
    else:
        raise ValueError('Expected image_file parameter to be either ImageFile or str')

    num_channels = len(img.getbands())
    channel_sums = np.zeros(num_channels+1, dtype=dtype)

    for i in range(0, num_channels):
        channel_sums[i] = sum(img.getdata(i))

    channel_sums[-1] = img.width * img.height

    if verbose:
        global _per_channel_mean_processed
        _per_channel_mean_processed.value += 1

        if _per_channel_mean_processed.value % 1000 == 0:
            print 'Per-channel mean: processed {} images'.format(_per_channel_mean_processed.value)

    return channel_sums


def calculate_per_channel_stddev(image_files, per_channel_mean, verbose=False, parallelization_backend='threading'):
    # type: (list[ImageFile], np.ndarray, bool, str) -> np.ndarray

    """
    Calculates the per-channel standard deviation from all the given ImageFiles or image paths and returns it as
    a N_CHANNELS dimensional numpy array.

    # Arguments
        :param image_files: list of ImageFiles or image paths to be used in the calculation
        :param per_channel_mean: per channel mean for the data set in range [-1,1]
        :param verbose: print information about the progress of the calculation
        :param parallelization_backend: which parallelization method to use 'threading' or 'multiprocessing'
    # Returns
        :return: numpy array with the channel means in range [-1, 1]
    """

    # Parallelize per-channel variance calculations
    n_jobs = get_number_of_parallel_jobs()

    channel_variances = Parallel(n_jobs=n_jobs, backend=parallelization_backend)(
        delayed(_get_per_channel_var)(
            image_file, per_channel_mean, np.float64, verbose) for image_file in image_files)

    # Calculate the final value
    channel_variances = sum(channel_variances)
    num_pixels = channel_variances[-1]
    channel_variances = channel_variances[:-1]

    # Calculate final variance value
    per_channel_var = channel_variances / num_pixels

    # Calculate the stddev
    per_channel_stddev = np.sqrt(per_channel_var)

    if verbose:
        global _per_channel_stddev_processed
        _per_channel_stddev_processed.value = 0

    return per_channel_stddev


def _get_per_channel_var(image_file, per_channel_mean, dtype=np.float64, verbose=False):
    # type: (ImageFile, np.ndarray, np.dtype, bool) -> np.ndarray

    if isinstance(image_file, ImageFile):
        img = image_file.get_image()
    elif isinstance(image_file, str):
        img = load_img(image_file)
    else:
        raise ValueError('Expected image_file parameter to be either ImageFile or str')

    np_img = img_to_array(img)
    num_channels = np_img.shape[-1]
    channel_variances = np.zeros(num_channels+1, dtype=dtype)

    # Normalize to [-1 ,1]
    np_img /= 255.0
    np_img -= 0.5
    np_img *= 2.0

    # Calculate variance
    np_img -= per_channel_mean
    np_img **= 2
    channel_variances[:-1] = np.sum(np_img, axis=(0, 1))

    # Append number of pixels
    channel_variances[-1] = img.width * img.height

    if verbose:
        global _per_channel_stddev_processed
        _per_channel_stddev_processed.value += 1

        if _per_channel_stddev_processed.value % 1000 == 0:
            print 'Per-channel stddev: processed {} images'.format(_per_channel_stddev_processed.value)

    return channel_variances


def calculate_material_samples_class_probabilities(material_samples):
    # type: (list[list[MaterialSample]]) -> list[float]

    class_probabilities = []

    for class_samples in material_samples:
        class_bbox_pixels = 0.0
        class_mat_pixels = 0.0

        for material_sample in class_samples:
            class_bbox_pixels += material_sample.bbox_size
            class_mat_pixels += material_sample.num_material_pixels

        class_probabilities.append(class_mat_pixels/max(class_bbox_pixels, 1.0))

    return class_probabilities


def calculate_material_samples_mfb_class_weights(material_samples):
    class_probabilities = np.array(calculate_material_samples_class_probabilities(material_samples))
    non_zero_class_probabilities = np.array([p for p in class_probabilities if p > 0.0])
    median_probability = np.median(non_zero_class_probabilities)

    # Calculate class weights and avoid division by zero for ignored classes such as
    # the background
    class_weights = []

    for p in class_probabilities:
        if p > 0.0:
            class_weights.append(median_probability/p)
        else:
            class_weights.append(0.0)

    return class_weights


def calculate_material_samples_enet_class_weights(material_samples):
    c = 1.02
    class_probabilities = np.array(calculate_material_samples_class_probabilities(material_samples))

    # Calculate class weights and avoid division by zero for ignored classes such as
    # the background
    class_weights = []

    for p in class_probabilities:
        if p > 0.0:
            class_weights.append(1.0/np.log(c+p))
        else:
            class_weights.append(0.0)

    return class_weights


def get_mask_statistics(mask_file_path, red_color_to_material_id, num_classes, class_frequencies_dtype=np.uint64):
    # type: (str, dict, int, np.dtype) -> (int, np.ndarray)

    # Load the red channel of the mask image
    img = load_img(mask_file_path)
    img = img.split()[0]

    num_pixels = img.width * img.height
    red_histogram = img.histogram()
    material_frequencies = np.zeros(num_classes, dtype=class_frequencies_dtype)

    for r_color, num_r_color_pixels in enumerate(red_histogram):
        if num_r_color_pixels != 0:
            material_id = red_color_to_material_id.get(r_color)

            if material_id is None:
                print 'WARNING: Could not find material id for red color {} with {} pixels in mask {}'.format(r_color, num_r_color_pixels, mask_file_path)
                continue

            material_frequencies[material_id] += num_r_color_pixels

    return num_pixels, material_frequencies


def get_material_samples(mask_files, material_class_information, background_class=0, min_sample_size=5, parallelization_backend='threading', verbose=False):
    """
    Given a set of mask files calculates the MaterialSamples in those mask files.
    Will return a list where each index contains a list of material samples for the
    respective material class. The list for background class will always be empty.

    # Arguments
        :param mask_files: a list of segmentation mask files as ImageFiles
        :param material_class_information:
        :param background_class: index of the background class, default 0
        :param min_sample_size: minimum pixels per MaterialSample to avoid degenerate sets due to artefacts in the mask images
        :param parallelization_backend: parallelization method, 'threading' or 'multiprocessing'
        :param verbose: print progress information
    # Returns
        :return: A list of material sample lists (24xN_SAMPLES_IN_CLASS)
    """
    # type: (list[ImageFile]) -> list[list[MaterialSample]]
    import itertools

    r_color_to_material_id = get_red_color_to_material_id_lookup_dict(material_class_information)

    # Parallelize the calculation for different files
    n_jobs = get_number_of_parallel_jobs()

    data = Parallel(n_jobs=n_jobs, backend=parallelization_backend)(
        delayed(_get_material_samples)(
            mask_file, r_color_to_material_id, background_class, min_sample_size, verbose) for mask_file in mask_files)

    data = list(itertools.chain.from_iterable(data))

    if verbose:
        global _material_samples_processed
        print 'Found in total {} material samples'.format(len(data))
        _material_samples_processed.value = 0

    # Order the material samples according to their material index (id)
    material_samples = [list() for _ in range(len(material_class_information))]

    for ms in data:
        material_samples[ms.material_id].append(ms)

    return material_samples


def _from_2d_to_1d_index(y, x, width):
    return y * width + x


def _get_material_samples(mask_file, r_color_to_material_id, background_class=0, min_sample_size=5, verbose=False):
    # type: (ImageFile, dict, int, int) -> list[MaterialSample]

    from collections import deque

    # Take only the red channel of the image
    pil_r_channel = mask_file.get_image().split()[0]
    width = pil_r_channel.width
    height = pil_r_channel.height

    r_channel = img_to_array(pil_r_channel, dtype=np.uint8).squeeze()
    pixel_set_references = [None] * (height * width)
    unique_pixel_sets = []
    queue = deque()

    for y in range(0, height):
        for x in range(0, width):

            # If this is a background pixel or already labeled - continue
            if r_channel[y][x] == background_class or pixel_set_references[_from_2d_to_1d_index(y, x, width)] is not None:
                continue

            # If this pixel is a foreground pixel and it is not already labeled
            # add it as the first element in a queue
            queue.append((r_channel[y][x], y, x))

            # print 'New set start: {},{}'.format(y, x)
            cur_px_set = list()
            unique_pixel_sets.append(cur_px_set)

            while queue:
                # Unwrap the pixel information and get a reference to the current set
                px_info = queue.pop()
                px_val, px_y, px_x = px_info

                cur_px_set.append(px_info)

                # Store the set reference to this pixel
                pixel_set_references[_from_2d_to_1d_index(px_y, px_x, width)] = cur_px_set

                # 8-connectivity neighbour check:
                # If a neighbour is a foreground pixel and is not already labelled,
                # give it the current label and add it to the queue.
                # Loop through the 8-neighbours
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue

                        ny = px_y + i
                        nx = px_x + j
                        n_val = r_channel[ny][nx] if (ny >= 0 and ny < height and nx >= 0 and nx < width) else background_class

                        if px_val == n_val and pixel_set_references[_from_2d_to_1d_index(ny, nx, width)] is None:
                            queue.append((n_val, ny, nx))
                            pixel_set_references[_from_2d_to_1d_index(ny, nx, width)] = cur_px_set

    # Filter sets smaller than min_set_size
    unique_pixel_sets = [s for s in unique_pixel_sets if len(s) >= min_sample_size]

    if verbose:
        global _material_samples_processed
        _material_samples_processed.value += 1

        if _material_samples_processed.value % 1000 == 0:
            print 'Material samples: {} images processed'.format(_material_samples_processed.value)
        #print 'Found {} unique pixel sets for file {}'.format(len(unique_pixel_sets), mask_file.file_name)

    # Build the material samples from the unique pixel sets
    material_samples = []

    for s in unique_pixel_sets:
        material_r_color = s[0][0]
        material_id = r_color_to_material_id[material_r_color]
        material_samples.append(MaterialSample(file_name=mask_file.file_name,
                                               material_id=material_id,
                                               material_r_color=material_r_color,
                                               image_width=width,
                                               image_height=height,
                                               pixel_info_list=s))

    return material_samples


def one_hot_encode_mask(np_mask_img, material_class_information, verbose=False):
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


def split_labeled_dataset(photo_files, mask_files, split, random_seed):
    # type: (list[str], list[str], list[float], int) -> (list[str], list[str], list[str])

    """
    Splits the whole dataset randomly into three different groups: training,
    validation and test, according to the split provided as the parameter.
    The provided photo and segmentation lists should have matching filenames so
    that after sorting the arrays have matching pairs in matching indices.

    # Arguments
        :param photo_files: photo files
        :param mask_files: segmentation mask files
        :param split: a list of floats describing the dataset split, must sum to one: [training, validation, test]
        :param random_seed: random seed
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
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.shuffle(photo_mask_files)

    # Divide the dataset to three different parts: training, validation and test
    # according to the given split: 0=training, 1=validation, 2=test
    dataset_size = len(photo_mask_files)
    training_set_size = int(round(split[0] * dataset_size))
    validation_set_size = int(round(split[1] * dataset_size))
    test_set_size = int(round(split[2] * dataset_size))

    # If the sizes don't match exactly add/subtract the different
    # from the training set
    total_size = training_set_size + validation_set_size + test_set_size

    if total_size != dataset_size:
        diff = dataset_size - (training_set_size + validation_set_size + test_set_size)
        training_set_size += diff

    total_size = training_set_size + validation_set_size + test_set_size

    if total_size != dataset_size:
        raise ValueError('The split set sizes do not sum to total dataset size: {} + {} + {} = {} != {}'
                         .format(training_set_size, validation_set_size, test_set_size, total_size, dataset_size))

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
    Counts the trailing zeroes in the binary of the parameter number.

    # Arguments
        :param num: the number
    # Returns
        :return: the number of trailing zeroes
    """

    zeroes = 0
    while ((num >> zeroes) & 1) == 0:
        zeroes = zeroes + 1
    return zeroes
