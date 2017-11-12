# coding=utf-8

import argparse
import time
import random
import numpy as np
import jsonpickle

from PIL import ImageFile

from .. import settings
from ..utils import dataset_utils
from ..utils.dataset_utils import SegmentationSetInformation, SegmentationTrainingSetInformation, SegmentationDataSetInformation
from ..data_set import ImageSet


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--photos", required=True, type=str, help="Path to the labeled photo files")
    ap.add_argument("-m", "--masks", required=True, type=str, help="Path to the labeled mask files")
    ap.add_argument("-u", "--unlabeled", required=False, type=str, help="Path to possible unlabeled data")
    ap.add_argument("-c", "--categories", required=True, type=str, help="Path to materials CSV file")
    ap.add_argument("-r", "--rseed", required=True, type=int, help="Random seed")
    ap.add_argument("-s", "--split", required=True, type=str, help="Dataset split e.g. '0.8,0.05,0.15'")
    ap.add_argument("--stats", required=False, type=bool, default=False, help="Calculate statistics for the new training set")
    ap.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    ap.add_argument("--maxjobs", required=False, type=int, help="Specify maximum number of parallel jobs")
    args = vars(ap.parse_args())

    photos_path = args["photos"]
    masks_path = args["masks"]
    unlabeled_path = args["unlabeled"]
    materials_path = args["categories"]
    rseed = args["rseed"]
    calculate_statistics = args["stats"]
    split = [float(v.strip()) for v in args["split"].split(',')]
    max_jobs = args["maxjobs"]
    output_path = args["output"]
    parallellization_backend = 'multiprocessing'

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print 'Using random seed: {}'.format(rseed)
    random.seed(rseed)
    np.random.seed(rseed)

    if max_jobs:
        print 'Setting maximum number of parallel jobs to: {}'.format(max_jobs)
        settings.MAX_NUMBER_OF_JOBS = max_jobs

    print 'Loading material information from file: {}'.format(materials_path)
    materials = dataset_utils.load_material_class_information(materials_path)
    print 'Loaded {} materials'.format(len(materials))

    print 'Reading photos from: {}'.format(photos_path)
    photo_image_set = ImageSet("labeled_photos", photos_path)
    print 'Found {} photo images'.format(photo_image_set.size)

    print 'Reading masks from: {}'.format(masks_path)
    mask_image_set = ImageSet("labeled_masks", masks_path)
    print 'Found {} mask images'.format(mask_image_set.size)

    print 'Splitting the dataset using split: {}'.format(split)
    training, validation, test = dataset_utils.split_labeled_dataset(photo_files=photo_image_set.image_files, mask_files=mask_image_set.image_files, split=split, random_seed=rseed)

    training_labeled_masks = [f[1] for f in training]
    validation_labeled_masks = [f[1] for f in validation]
    test_labeled_masks = [f[1] for f in test]
    training_unlabeled_photos = []

    if unlabeled_path:
        print 'Reading unlabeled photos from directory: {}'.format(unlabeled_path)
        unlabeled_image_set = ImageSet("unlabeled_photos", unlabeled_path)
        print 'Found {} unlabeled photo images'.format(unlabeled_image_set.size)
        training_unlabeled_photos = unlabeled_image_set.image_files

    training_labeled_photos_file_names = [f[0].file_name for f in training]
    training_labeled_masks_file_names = [f[1].file_name for f in training]
    training_unlabeled_photos_file_names = [f.file_name for f in training_unlabeled_photos] if len(training_unlabeled_photos) > 0 else []
    validation_labeled_photos_file_names = [f[0].file_name for f in validation]
    validation_labeled_masks_file_names = [f[1].file_name for f in validation]
    test_labeled_photos_file_names = [f[0].file_name for f in test]
    test_labeled_masks_file_names = [f[1].file_name for f in test]

    print 'Calculating material samples for training set'
    s_time = time.time()
    training_material_samples = dataset_utils.get_material_samples(mask_files=training_labeled_masks, material_class_information=materials, parallelization_backend=parallellization_backend, verbose=True)
    print 'Training set material samples calculation finished in: {}s'.format(time.time()-s_time)

    print 'Calculating material samples for validation set'
    s_time = time.time()
    validation_material_samples = dataset_utils.get_material_samples(mask_files=validation_labeled_masks, material_class_information=materials, parallelization_backend=parallellization_backend, verbose=True)
    print 'Validation set material samples calculation finished in: {}s'.format(time.time()-s_time)

    print 'Calculating material samples for test set'
    s_time = time.time()
    test_material_samples = dataset_utils.get_material_samples(mask_files=test_labeled_masks, material_class_information=materials, parallelization_backend=parallellization_backend, verbose=True)
    print 'Test set material samples calculation finished in: {}s'.format(time.time()-s_time)

    training_set = SegmentationTrainingSetInformation('training set', training_labeled_photos_file_names, training_labeled_masks_file_names, training_unlabeled_photos_file_names, training_material_samples)
    validation_set = SegmentationSetInformation('validation set', validation_labeled_photos_file_names, validation_labeled_masks_file_names, validation_material_samples)
    test_set = SegmentationSetInformation('test set', test_labeled_photos_file_names, test_labeled_masks_file_names, test_material_samples)
    data_set = SegmentationDataSetInformation(training_set, validation_set, test_set, rseed)

    if calculate_statistics:
        print 'Starting statistics calculation for the data set'

        data_set.load_statistics(labeled_photos_folder_path=photos_path,
                                 unlabeled_photos_folder_path=unlabeled_path,
                                 masks_folder_path=masks_path,
                                 material_class_information=materials,
                                 parallelization_backend=parallellization_backend,
                                 verbose=True)

    print 'Encoding data set information to JSON'
    json_str = jsonpickle.encode(data_set)
    print 'Writing data set information to: {}'.format(output_path)

    with open(output_path, 'w') as f:
        f.write(json_str)

if __name__ == '__main__':
    main()
