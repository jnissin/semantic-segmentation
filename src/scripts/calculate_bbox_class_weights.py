# coding=utf-8

import argparse
import numpy as np

from ..utils import dataset_utils


def main():

    ap = argparse.ArgumentParser(description="Calculates class weights from the bbox data of a data set information file")
    ap.add_argument("-f", "--file", required=True, help="Path to data set information file")
    ap.add_argument("-m", "--mode", required=False, default="mfb", choices=['mfb', 'enet'])
    args = vars(ap.parse_args())

    path_to_data_set_information_file = args['file']
    mode = args['mode']

    print 'Loading data set information from: {}'.format(path_to_data_set_information_file)
    data_set_information = dataset_utils.load_segmentation_data_set_information(path_to_data_set_information_file)
    print 'Loaded data set information successfully with set sizes (tr,va,te): {}, {}, {}'\
        .format(data_set_information.training_set.labeled_size,
                data_set_information.validation_set.labeled_size,
                data_set_information.test_set.labeled_size)

    material_samples = data_set_information.test_set.material_samples
    num_material_classes = len(material_samples)
    num_material_samples = sum([len(class_samples) for class_samples in material_samples])

    print 'Calculating {} weights from {} material samples in {} material classes from the training set'.format(mode, num_material_samples, num_material_classes)

    if mode == 'mfb':
        class_weights = dataset_utils.calculate_material_samples_mfb_class_weights(material_samples)
    elif mode == 'enet':
        class_weights = dataset_utils.calculate_material_samples_enet_class_weights(material_samples)
    else:
        raise ValueError('Uknown mode: {}'.format(mode))

    non_zero_class_weights = [w for w in class_weights if w > 0.0]
    print 'Weights: {}'.format(class_weights)
    print 'Weights (excl. 0.0): min: {}, max: {}, range: {}, variance: {}, stddev: {}'.format(np.min(non_zero_class_weights),
                                                                                             np.max(non_zero_class_weights),
                                                                                             np.max(non_zero_class_weights) - np.min(non_zero_class_weights),
                                                                                             np.var(non_zero_class_weights),
                                                                                             np.std(non_zero_class_weights))
    print 'Done'


if __name__ == '__main__':
    main()
