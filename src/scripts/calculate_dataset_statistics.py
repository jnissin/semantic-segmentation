# coding=utf-8

import argparse
import os
import time
import json
import random
import multiprocessing
import numpy as np
from PIL import ImageFile

from keras.preprocessing.image import list_pictures, load_img, img_to_array
from joblib import Parallel, delayed

from ..utils import dataset_utils


class DatasetStatistics(object):

    def __init__(self,
                 mask_statistics,
                 per_channel_mean,
                 per_channel_stddev,
                 rseed,
                 split):
        # type: (list[MaskStatistics], list[float], list[float], int, list[float]) -> ()

        self.mask_statistics = mask_statistics
        self.rseed = rseed
        self.split = split
        self.per_channel_mean = per_channel_mean
        self.per_channel_stddev = per_channel_stddev

        # Calculate total number of pixels
        self.total_mask_pixels = sum([s.num_pixels for s in self.mask_statistics])

        # Calculate total number of pixels of different materials
        self.material_pixels = [s.material_pixels for s in self.mask_statistics]
        self.material_pixels = zip(*self.material_pixels)
        self.material_pixels = [sum(x) for x in self.material_pixels]

        # Calculate total number of occurences of different materials and record
        # the files that have instances of each material
        self.material_occurrences = [0] * len(self.material_pixels)
        self.material_occurrence_files = [[] for i in range(len(self.material_pixels))]

        for s in self.mask_statistics:
            for i in range(0, len(s.material_pixels)):
                if s.material_pixels[i] != 0:
                    self.material_occurrences[i] += 1
                    self.material_occurrence_files[i].append(s.image_name)

        # Calculate median frequency balancing weights. The frequency of a material freq(c)
        # is the number of pixels of class c divided by the total number of pixels in images
        # where c is present. Median freq is the median of these frequencies. If a material is not
        # present in the training set it will have a zero weight.
        total_pixels_per_material = [0] * len(self.material_pixels)

        for s in self.mask_statistics:
            for i in range(0, len(s.material_pixels)):
                if s.material_pixels[i] > 0:
                    total_pixels_per_material[i] += s.num_pixels

        np_material_pixels = np.array(self.material_pixels)
        np_total_pixels_per_material = np.array(total_pixels_per_material)

        # Avoid NaNs because of zero material frequencies
        np_material_frequencies = np.zeros_like(np_material_pixels)

        for i in range(0, np_total_pixels_per_material.shape[0]):
            if np_total_pixels_per_material[i] != 0:
                np_material_frequencies[i] = np_material_pixels[i] / np_total_pixels_per_material[i]

        # Take the median frequency
        np_median_material_frequency = np.median(np_material_frequencies)

        # Avoid NaNs because of zero material frequencies
        np_median_material_frequency_weights = np.zeros_like(np_material_frequencies)

        for i in range(0, np_material_frequencies.shape[0]):
            if np_material_frequencies[i] != 0:
                np_median_material_frequency_weights[i] = np_median_material_frequency / np_material_frequencies[i]

        self.median_frequency_balancing_weights = np_median_material_frequency_weights.tolist()


class MaskStatistics(object):

    def __init__(self, image_name, num_pixels, material_pixels):
        # type: (str, int, list[int]) -> ()
        self.image_name = image_name
        self.num_pixels = num_pixels
        self.material_pixels = material_pixels


def calculate_mask_statistics(mask_file_path, materials):
    # type: (str, list[MaterialClassInformation]) -> MaskStatistics

    mask_img = load_img(mask_file_path)
    mask_img_array = img_to_array(mask_img)
    expanded_mask = dataset_utils.expand_mask(mask_img_array, materials)

    image_name = os.path.basename(mask_file_path)
    num_pixels = mask_img_array.shape[0] * mask_img_array.shape[1]
    material_pixels = []

    for i in range(0, expanded_mask.shape[-1]):
        material_pixels.append(float(np.sum(expanded_mask[:, :, i])))

    img_stats = MaskStatistics(image_name, num_pixels, material_pixels)
    return img_stats


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--photos", required=True, type=str, help="Path to the labeled photo files")
    ap.add_argument("-m", "--masks", required=True, type=str, help="Path to the labeled mask files")
    ap.add_argument("-u", "--unlabeled", required=False, type=str, help="Path to possible unlabeled data")
    ap.add_argument("-c", "--categories", required=True, type=str, help="Path to materials CSV file")
    ap.add_argument("-r", "--rseed", required=True, type=int, help="Random seed")
    ap.add_argument("-s", "--split", required=True, type=str, help="Dataset split")
    ap.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    args = vars(ap.parse_args())

    photos_path = args["photos"]
    masks_path = args["masks"]
    unlabeled_path = args["unlabeled"]
    materials_path = args["categories"]
    rseed = args["rseed"]
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

    print 'Reading masks from directory: {}'.format(masks_path)
    masks = list_pictures(masks_path)
    print 'Found {} mask images'.format(len(masks))

    print 'Reading photos from directory: {}'.format(photos_path)
    photos = list_pictures(photos_path)
    print 'Found {} photo images'.format(len(photos))

    print 'Splitting the dataset using: {}'.format(split)
    training, validation, test = dataset_utils.split_labeled_dataset(photo_files=photos, mask_files=masks, split=split)

    photos = [pair[0] for pair in training]
    masks = [pair[1] for pair in training]

    unlabeled_photos = []

    if unlabeled_path:
        print 'Reading unlabeled photos from directory: {}'.format(unlabeled_path)
        unlabeled_photos = list_pictures(unlabeled_path)
        print 'Found {} unlabeled photo images'.format(len(unlabeled_photos))

    photos = photos + unlabeled_photos

    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    print 'Starting mask statistics calculation for {} masks with {} jobs'.format(len(masks), n_jobs)

    start_time = time.time()
    mask_stats = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(calculate_mask_statistics)(f, materials) for f in masks)

    print 'Mask statistics calculation of {} files finished in {} seconds'\
        .format(len(mask_stats), time.time()-start_time)

    print 'Starting per-channel mean calculation for {} photos with {} jobs'.format(len(photos), n_jobs)

    start_time = time.time()
    per_channel_mean = dataset_utils.calculate_per_channel_mean(photos, 3, verbose=True)

    print 'Per-channel mean calculation for {} files finished in {} seconds'\
        .format(len(photos), time.time()-start_time)

    print 'Starting per-channel stddev calculation for {} photos with {} jobs'.format(len(photos), n_jobs)

    per_channel_stddev = dataset_utils.calculate_per_channel_stddev(photos, per_channel_mean, 3, verbose=True)

    print 'Per-channel stddev calculation for {} files finished in {} seconds'\
        .format(len(photos), time.time()-start_time)

    dataset_stats = DatasetStatistics(mask_stats, per_channel_mean.tolist(), per_channel_stddev.tolist(), rseed, split)

    print 'Median frequency balancing weights: {}'.format(dataset_stats.median_frequency_balancing_weights)

    print 'Writing dataset statistics to file: {}'.format(output_path)
    json_str = json.dumps(dataset_stats, default=lambda o: o.__dict__)

    with open(output_path, 'w') as f:
        f.write(json_str)

    print 'Material statistics stored successfully'


if __name__ == '__main__':
    main()
