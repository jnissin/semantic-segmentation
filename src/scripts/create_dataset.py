# coding=utf-8

import argparse
import time
import random
import multiprocessing
import numpy as np
from PIL import ImageFile
import jsonpickle

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
    ap.add_argument("-s", "--split", required=True, type=str, help="Dataset split")
    ap.add_argument("--stats", required=False, type=bool, default=False, help="Calculate statistics for the new training set")
    ap.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    args = vars(ap.parse_args())

    photos_path = args["photos"]
    masks_path = args["masks"]
    unlabeled_path = args["unlabeled"]
    materials_path = args["categories"]
    rseed = args["rseed"]
    calculate_statistics = args["stats"]
    split = [float(v.strip()) for v in args["split"].split(',')]
    output_path = args["output"]

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print 'Using random seed: {}'.format(rseed)
    random.seed(rseed)
    np.random.seed(rseed)

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
    training, validation, test = dataset_utils.split_labeled_dataset(photo_files=photo_image_set.image_files,
                                                                     mask_files=mask_image_set.image_files,
                                                                     split=split)

    training_labeled_photos = [f[0] for f in training]
    training_labeled_masks = [f[1] for f in training]
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

    training_set = SegmentationTrainingSetInformation(training_labeled_photos_file_names, training_labeled_masks_file_names, training_unlabeled_photos_file_names)
    validation_set = SegmentationSetInformation(validation_labeled_photos_file_names, validation_labeled_masks_file_names)
    test_set = SegmentationSetInformation(test_labeled_photos_file_names, test_labeled_masks_file_names)

    per_channel_mean = []
    per_channel_stddev = []
    class_weights = []

    if calculate_statistics:
        print 'Starting statistics calculation for the training set'

        num_cores = multiprocessing.cpu_count()
        n_jobs = min(32, num_cores)

        if_photos = training_labeled_photos + training_unlabeled_photos
        if_masks = training_labeled_masks

        print 'Starting per-channel mean calculation for {} photos with {} jobs'.format(len(if_photos), n_jobs)
        start_time = time.time()
        per_channel_mean = dataset_utils.calculate_per_channel_mean(if_photos, 3, verbose=True).tolist()
        print 'Per-channel mean calculation for {} files finished in {} seconds'.format(len(if_photos), time.time()-start_time)

        print 'Starting per-channel stddev calculation for {} photos with {} jobs'.format(len(if_photos), n_jobs)
        start_time = time.time()
        per_channel_stddev = dataset_utils.calculate_per_channel_stddev(if_photos, per_channel_mean, 3, verbose=True).tolist()
        print 'Per-channel stddev calculation for {} files finished in {} seconds'.format(len(if_photos), time.time()-start_time)

        print 'Starting median frequency balancing weights calculation for {} masks with {} jobs'.format(len(if_masks), n_jobs)
        start_time = time.time()
        class_weights = dataset_utils.calculate_median_frequency_balancing_weights(if_masks, material_class_information=materials).tolist()
        print 'Median frequency balancing weight calculation for {} files finished in {} seconds'.format(len(if_masks), time.time()-start_time)

    data_set = SegmentationDataSetInformation(training_set,
                                              validation_set,
                                              test_set,
                                              rseed,
                                              per_channel_mean,
                                              per_channel_stddev,
                                              class_weights)

    json_str = jsonpickle.encode(data_set)

    print 'Writing data set information to: {}'.format(output_path)

    with open(output_path, 'w') as f:
        f.write(json_str)

if __name__ == '__main__':
    main()
