# coding=utf-8

import argparse
import random

import numpy as np


class PatchSample(object):

    def __init__(self, label, photo_id, x, y):
        # type: (int, str, float, float) -> None

        self.label = label
        self.photo_id = photo_id
        self.x = x
        self.y = y


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--trainingset", required=True, type=str, help="Path to the MINC training set file")
    ap.add_argument("-o", "--output", required=True, type=str, help="Path to the new balanced MINC training set file (output)")
    ap.add_argument("-s", "--seed", required=True, type=int, help="Random seed for the selection of samples")
    args = vars(ap.parse_args())

    file_path = args['trainingset']
    output_path = args['output']
    random_seed = args['seed']

    # Parse the training set file
    # Each line is a 4-tuple list of (label, photo id, x, y)
    print 'Reading training set from file: {}'.format(file_path)

    with open(file_path) as f:
        lines = f.readlines()

    total_patch_samples = 0
    patch_samples_per_category = {}

    for line in lines:
        line = line.strip()

        # Filter any empty lines
        if len(line) == 0:
            continue

        parts = line.split(',')

        if len(parts) != 4:
            raise ValueError('Invalid data, expected 4 parts got: {}'.format(len(parts)))

        label = int(parts[0].strip())
        photo_id = parts[1].strip()
        x = float(parts[2])
        y = float(parts[3])

        if label not in patch_samples_per_category:
            patch_samples_per_category[label] = []

        patch_samples_per_category[label].append(PatchSample(label=label, photo_id=photo_id, x=x, y=y))
        total_patch_samples += 1

    # Calculate category sizes
    size_per_category = [len(patch_samples_per_category[key]) for key in patch_samples_per_category.keys()]

    # Print statistics and find the minimum class
    print 'Found {} patch samples in {} different categories'.format(total_patch_samples, len(patch_samples_per_category.keys()))
    print 'Samples per category:'

    for label, size in enumerate(size_per_category):
        print 'Category {}: {}'.format(label, size)

    # Find the minimum class
    min_val, min_idx = min((val, idx) for (idx, val) in enumerate(size_per_category))

    # Find the maximum class
    max_val, max_idx = max((val, idx) for (idx, val) in enumerate(size_per_category))

    print 'Minimum class {}: {} samples'.format(min_idx, min_val)
    print 'Maximum class {}: {} samples'.format(max_idx, max_val)

    # Select the same number of samples from each class (according to min_val)
    print 'Selecting {} samples from each class using random seed: {}'.format(min_val, random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    balanced_patch_samples_per_category = {}

    for key, val in patch_samples_per_category.iteritems():
        selected_patch_indices = np.random.permutation(len(val))[0:min_val]
        selected_patches = [val[patch_idx] for patch_idx in selected_patch_indices]
        balanced_patch_samples_per_category[key] = selected_patches

        if len(balanced_patch_samples_per_category[key]) != min_val:
            raise ValueError('Something went wrong during sample selection size does not match the minimum size: {} vs {}'
                             .format(min_val, len(balanced_patch_samples_per_category[key])))

    # Create the new file
    print 'Writing the new balanced data set to file: {}'.format(output_path)

    with open(output_path, 'w') as output_file:
        for key, val in balanced_patch_samples_per_category.iteritems():
            print 'Writing category: {}'.format(key)

            for patch_sample in val:
                output_file.write('{},{},{},{}\n'.format(patch_sample.label, patch_sample.photo_id, patch_sample.x, patch_sample.y))

    print 'Done'

if __name__ == '__main__':
    main()


