# coding=utf-8

import argparse
import os
import time
import json
import jsonpickle
import multiprocessing
import numpy as np
from PIL import ImageFile

from keras.preprocessing.image import img_to_array
from joblib import Parallel, delayed

from ..utils import dataset_utils
from ..utils.dataset_utils import SegmentationDataSetInformation, MaterialClassInformation
from ..data_set import ImageSet, ImageFile


class MaterialStatistics(object):

    def __init__(self, image_name, image_size, material_pixels):
        # type: (str, tuple[int], list[int]) -> ()

        self.image_name = image_name
        self.image_size = image_size
        self.material_pixels = material_pixels

        self.num_pixels = image_size[0]*image_size[1]


class PhotoStatistics(object):

    def __init__(self, image_name, image_size):
        # type: (str, tuple[int]) -> ()

        self.image_name = image_name
        self.image_size = image_size
        self.num_pixels = image_size[0]*image_size[1]


class DataSetStatistics(object):

    def __init__(self,
                 materials,
                 total_material_statistics,
                 total_photo_statistics,
                 data_set_information=None):
        # type: (list[str], list[MaterialStatistics], list[PhotoStatistics], SegmentationDataSetInformation) -> ()

        self.materials = materials
        self.total_data_set_statistics = SetStatistics(total_material_statistics, total_photo_statistics)

        self.training_set_statistics = None
        self.validation_set_statistics = None
        self.test_set_statistics = None

        # If we have data set splits, calculate the statistics for each subset
        if data_set_information is not None:

            # Training set statistics
            training_set_labeled_photo_names = set(data_set_information.training_set.labeled_photos)
            training_set_unlabeled_photo_names = set(data_set_information.training_set.unlabeled_photos)
            training_set_labeled_mask_names = set(data_set_information.training_set.labeled_masks)
            training_set_labeled_photo_statistics = [ps for ps in total_photo_statistics if ps.image_name in training_set_labeled_photo_names]
            training_set_unlabeled_photo_statistics = [ps for ps in total_photo_statistics if ps.image_name in training_set_unlabeled_photo_names]
            training_set_photo_statistics = training_set_labeled_photo_statistics + training_set_unlabeled_photo_statistics
            training_set_material_statistics = [ms for ms in total_material_statistics if ms.image_name in training_set_labeled_mask_names]

            if len(training_set_photo_statistics) != data_set_information.training_set.labeled_size + data_set_information.training_set.unlabeled_size or \
                            len(training_set_material_statistics) != data_set_information.training_set.labeled_size:
                raise ValueError('Training set data sizes do not match')

            self.training_set_statistics = SetStatistics(training_set_material_statistics, training_set_photo_statistics)

            # Validation set statistics
            validation_set_labeled_photo_names = set(data_set_information.validation_set.labeled_photos)
            validation_set_labeled_mask_names = set(data_set_information.validation_set.labeled_masks)
            validation_set_photo_statistics = [ps for ps in total_photo_statistics if ps.image_name in validation_set_labeled_photo_names]
            validation_set_material_statistics = [ms for ms in total_material_statistics if ms.image_name in validation_set_labeled_mask_names]

            if len(validation_set_photo_statistics) != data_set_information.validation_set.labeled_size or \
                            len(validation_set_material_statistics) != data_set_information.validation_set.labeled_size:
                raise ValueError('Validation set data sizes do not match')

            self.validation_set_statistics = SetStatistics(validation_set_material_statistics, validation_set_photo_statistics)

            # Test set statistics
            test_set_labeled_photo_names = set(data_set_information.test_set.labeled_photos)
            test_set_labeled_mask_names = set(data_set_information.test_set.labeled_masks)
            test_set_photo_statistics = [ps for ps in total_photo_statistics if ps.image_name in test_set_labeled_photo_names]
            test_set_material_statistics = [ms for ms in total_material_statistics if ms.image_name in test_set_labeled_mask_names]

            if len(test_set_photo_statistics) != data_set_information.test_set.labeled_size or \
                            len(test_set_material_statistics) != data_set_information.test_set.labeled_size:
                raise ValueError('Test set data sizes do not match')

            self.test_set_statistics = SetStatistics(test_set_material_statistics, test_set_photo_statistics)


class SetStatistics(object):

    def __init__(self,
                 material_statistics,
                 photo_statistics):
        # type: (list[MaterialStatistics], list[PhotoStatistics]) -> ()

        self.num_labeled_data = len(material_statistics)
        self.num_unlabeled_data = len(photo_statistics) - self.num_labeled_data

        self.material_statistics = material_statistics

        # Calculate total number of labeled pixels
        self.total_labeled_pixels = SetStatistics._calculate_total_number_of_labeled_pixels(self.material_statistics)
        self.total_unlabeled_pixels = SetStatistics._calculate_total_number_of_pixels(photo_statistics) - self.total_labeled_pixels

        # Calculate total number of pixels of different materials
        self.material_pixels = SetStatistics._calculate_material_pixels(self.material_statistics)

        # Calculate total number of occurences of different materials and record
        # the files that have instances of each material
        self.material_occurrences, self.material_occurrence_files = SetStatistics._calculate_material_occurence_statistics(self.material_statistics, self.material_pixels)

        # Calculate image sizes
        self.image_sizes = SetStatistics._calculate_image_sizes(photo_statistics)

    @staticmethod
    def _calculate_total_number_of_labeled_pixels(material_statistics):
        # type: (list[MaterialStatistics]) -> int

        return sum([s.num_pixels for s in material_statistics])

    @staticmethod
    def _calculate_total_number_of_pixels(photo_statistics):
        # type: (list[PhotoStatistics]) -> int
        return sum([p.num_pixels for p in photo_statistics])

    @staticmethod
    def _calculate_material_pixels(material_statistics):
        # type: (list[MaterialStatistics]) -> list[int]

        material_pixels = [s.material_pixels for s in material_statistics]
        material_pixels = zip(*material_pixels)
        material_pixels = [sum(x) for x in material_pixels]
        return material_pixels

    @staticmethod
    def _calculate_material_occurence_statistics(material_statistics, material_pixels):
        # type: (list[MaterialStatistics], list[int]) -> (list[int], list[list[str]])

        material_occurrences = [0] * len(material_pixels)
        material_occurrence_files = [[] for i in range(len(material_pixels))]

        for s in material_statistics:
            for i in range(0, len(s.material_pixels)):
                if s.material_pixels[i] != 0:
                    material_occurrences[i] += 1
                    material_occurrence_files[i].append(s.image_name)

        return material_occurrences, material_occurrence_files

    @staticmethod
    def _calculate_image_sizes(photo_statistics):
        # type: (list[PhotoStatistics]) -> dict[tuple[int], int]
        sizes = dict()

        for p in photo_statistics:
            if sizes.get(p.image_size):
                sizes[p.image_size] += 1
            else:
                sizes[p.image_size] = 1

        return sizes


def calculate_material_statistics(mask_img_file, materials):
    # type: (ImageFile, list[MaterialClassInformation]) -> MaterialStatistics

    mask_img = mask_img_file.get_image()
    mask_img_array = img_to_array(mask_img)
    expanded_mask = dataset_utils.expand_mask(mask_img_array, materials)

    image_name = mask_img_file.file_name
    material_pixels = []

    for i in range(0, expanded_mask.shape[-1]):
        material_pixels.append(float(np.sum(expanded_mask[:, :, i])))

    img_stats = MaterialStatistics(image_name, mask_img.size, material_pixels)
    return img_stats


def calculate_photo_statistics(photo_img_file):
    # type: (ImageFile) -> PhotoStatistics

    photo_img = photo_img_file.get_image()
    image_name = photo_img_file.file_name
    photo_stats = PhotoStatistics(image_name=image_name, image_size=photo_img.size)
    return photo_stats


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--photos", required=True, type=str, help="Path to the labeled photo files")
    ap.add_argument("-m", "--masks", required=True, type=str, help="Path to the labeled mask files")
    ap.add_argument("-u", "--unlabeled", required=False, type=str, help="Path to possible unlabeled data")
    ap.add_argument("-c", "--categories", required=True, type=str, help="Path to materials CSV file")
    ap.add_argument("-d", "--dataset", required=False, type=str, help="Path to the data set .json file")
    ap.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    args = vars(ap.parse_args())

    photos_path = args["photos"]
    masks_path = args["masks"]
    unlabeled_path = args["unlabeled"]
    materials_path = args["categories"]
    path_to_data_set_information_file = args["dataset"]
    output_path = args["output"]

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print 'Loading material information from file: {}'.format(materials_path)
    materials = dataset_utils.load_material_class_information(materials_path)
    print 'Loaded {} materials'.format(len(materials))

    print 'Reading masks from directory: {}'.format(masks_path)
    masks = ImageSet(name="masks", path_to_archive=masks_path)
    print 'Found {} mask images'.format(masks.size)

    print 'Reading photos from directory: {}'.format(photos_path)
    photos = ImageSet(name="labeled photos", path_to_archive=photos_path)
    print 'Found {} photo images'.format(photos.size)

    unlabeled_photos = None

    if unlabeled_path:
        print 'Reading unlabeled photos from directory: {}'.format(unlabeled_path)
        unlabeled_photos = ImageSet(name="unlabeled photos", path_to_archive=unlabeled_path)
        print 'Found {} unlabeled photo images'.format(unlabeled_photos.size)

    photos_image_files = photos.image_files

    if unlabeled_photos:
        photos_image_files = photos.image_files + unlabeled_photos.image_files

    # Read data set information file if provided
    data_set_information = None

    if path_to_data_set_information_file:
        print 'Reading data set information from: {}'.format(path_to_data_set_information_file)
        data_set_information = dataset_utils.load_segmentation_data_set_information(path_to_data_set_information_file)

    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    # Calculate material statistics for all masks in the data set
    print 'Starting material statistics calculation for {} masks with {} jobs'.format(masks.size, n_jobs)
    start_time = time.time()
    mask_stats = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(calculate_material_statistics)(f, materials) for f in masks.image_files)
    print 'Material statistics calculation of {} files finished in {} seconds'.format(len(mask_stats), time.time()-start_time)

    # Calculate photo statistics for all photos in the data set
    print 'Starting photo statistics calculation for {} photos with {} jobs'.format(photos.size, n_jobs)
    start_time = time.time()
    photo_stats = Parallel(n_jobs, backend='multiprocessing')(delayed(calculate_photo_statistics)(f) for f in photos_image_files)
    print 'Photo statistics calculation of {} files finished in {} seconds'.format(len(photo_stats), time.time()-start_time)

    data_set_materials = [m.name for m in materials]
    data_set_stats = DataSetStatistics(data_set_materials, mask_stats, photo_stats, data_set_information)

    print 'Writing dataset statistics to file: {}'.format(output_path)
    #json_str = json.dumps(data_set_stats, default=lambda o: o.__dict__)
    json_str = jsonpickle.encode(data_set_stats)

    with open(output_path, 'w') as f:
        f.write(json_str)

    print 'Material statistics stored successfully'


if __name__ == '__main__':
    main()
