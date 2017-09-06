# coding=utf-8

import argparse
import numpy as np

from ..utils import dataset_utils

from ..utils.dataset_utils import MINCSample
from ..data_set import ImageSet


def _read_minc_data_set_file(file_path):
    data_set = list()

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            # Each line of the file is: 4-tuple list of (label, photo id, x, y)
            label, photo_id, x, y = line.split(',')
            data_set.append(MINCSample(minc_label=int(label), file_name=photo_id.strip(), x=float(x), y=float(y)))

    return data_set


def _read_minc_labels_translation_file(file_path):
    minc_to_custom_label_mapping = dict()

    with open(file_path, 'r') as f:
        # The first line should be skipped because it describes the data, which is
        # in the format of substance_name,minc_class_idx,custom_class_idx
        lines = f.readlines()

        for idx, line in enumerate(lines):
            # Skip the first line
            if idx == 0:
                continue

            substance_name, minc_class_idx, custom_class_idx = line.split(',')

            # Check that there are no duplicate entries for MINC class ids
            if minc_to_custom_label_mapping.has_key(int(minc_class_idx)):
                raise ValueError('Label mapping already contains entry for MINC class id: {}'.format(int(minc_class_idx)))

            # Check that there are no duplicate entries for custom class ids
            if int(custom_class_idx) in minc_to_custom_label_mapping.values():
                raise ValueError('Label mapping already contains entry for custom class id: {}'.format(int(custom_class_idx)))

            minc_to_custom_label_mapping[int(minc_class_idx)] = int(custom_class_idx)

    return minc_to_custom_label_mapping


def calculate_mfb_class_weights(class_probabilities):
    class_probabilities = np.array(class_probabilities)
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


def calculate_enet_class_weights(class_probabilities):
    c = 1.02
    class_probabilities = np.array(class_probabilities)

    # Calculate class weights and avoid division by zero for ignored classes such as
    # the background
    class_weights = []

    for p in class_probabilities:
        if p > 0.0:
            class_weights.append(1.0/np.log(c+p))
        else:
            class_weights.append(0.0)

    return class_weights


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--photos", required=True, help="Path to MINC photos folder")
    ap.add_argument("-d", "--dataset", required=True, help="Path to data set file")
    ap.add_argument("-l", "--labels", required=True, help="Path to the labels mapping CSV file")
    args = vars(ap.parse_args())

    photos_folder_path = args['photos']
    data_set_file_path = args['dataset']
    labels_mapping_file_path = args['labels']

    print 'Reading data set from file: {}'.format(data_set_file_path)
    minc_data_set = _read_minc_data_set_file(data_set_file_path)
    print 'Successfully read data set with {} samples'.format(len(minc_data_set))

    print 'Reading labels translations from file: {}'.format(labels_mapping_file_path)
    minc_to_custom_labels = _read_minc_labels_translation_file(labels_mapping_file_path)
    print 'Successfully read {} label mappings'.format(len(minc_to_custom_labels))

    print 'Constructing file filter list of necessary photo files for data set'
    data_set_photo_file_names = set([(sample.photo_id + '.jpg').lower() for sample in minc_data_set])
    print 'Data set references {} unique photo files'.format(len(data_set_photo_file_names))

    print 'Reading photo files from: {}'.format(photos_folder_path)
    image_set = ImageSet('data_set', photos_folder_path, list(data_set_photo_file_names))
    print 'Successfully constructed image set of {} files'.format(image_set.size)

    print 'Calculating per-channel mean'
    #per_channel_mean = dataset_utils.calculate_per_channel_mean(image_set.image_files, 3)
    per_channel_mean = np.array([0.041798265141623728, -0.059778711338833487, -0.17326070160306084])
    print 'Per channel mean calculation complete: {}'.format(list(per_channel_mean))

    print 'Calculating per-channel stddev'
    per_channel_stddev = dataset_utils.calculate_per_channel_stddev(image_set.image_files, per_channel_mean, 3)
    print 'Per channel stddev calculation complete: {}'.format(list(per_channel_stddev))

    print 'Calculating class frequencies in data set (custom labels)'
    class_labels = [minc_to_custom_labels[sample.minc_label] for sample in minc_data_set]
    num_classes = len(minc_to_custom_labels)
    class_frequencies = [0]*num_classes

    for val in class_labels:
        if val < 0 or val >= num_classes:
            raise ValueError('Invalid class label: {} with {} classes'.format(val, num_classes))
        class_frequencies[val] += 1

    print 'Class frequencies: {}'.format(class_frequencies)

    class_probabilities = np.array(class_frequencies).astype(np.float32)
    class_probabilities = class_probabilities / float(len(minc_data_set))
    print 'Class probabilities: {}'.format(list(class_probabilities))

    mfb_class_weights = calculate_mfb_class_weights(class_probabilities)
    print 'MFB class weights: {}'.format(list(mfb_class_weights))

    enet_class_weights = calculate_enet_class_weights(class_probabilities)
    print 'ENET class weights: {}'.format(list(enet_class_weights))

if __name__ == "__main__":
    main()
