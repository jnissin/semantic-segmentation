
import argparse
import numpy as np
from ..utils import dataset_utils


def calculate_total_pixels_per_class(material_samples):
    pixels_per_class = []

    for class_samples in material_samples:
        pixels_in_class = 0

        for sample in class_samples:
            pixels_in_class += sample.num_material_pixels

        pixels_per_class.append(pixels_in_class)

    return np.array(pixels_per_class)


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description="Calculates class weights from the bbox data of a data set information file")
    ap.add_argument("-f", "--file", required=True, help="Path to data set information file")
    args = vars(ap.parse_args())

    path_to_data_set_information_file = args['file']

    print 'Loading data set information from: {}'.format(path_to_data_set_information_file)
    data_set_information = dataset_utils.load_segmentation_data_set_information(path_to_data_set_information_file)
    print 'Loaded data set information successfully with set sizes (tr,va,te): {}, {}, {}'\
        .format(data_set_information.training_set.labeled_size,
                data_set_information.validation_set.labeled_size,
                data_set_information.test_set.labeled_size)

    print 'Whole data set'
    total_samples_per_class_tr = np.array([len(class_samples) for class_samples in data_set_information.training_set.material_samples])
    total_samples_per_class_val = np.array([len(class_samples) for class_samples in data_set_information.validation_set.material_samples])
    total_samples_per_class_te = np.array([len(class_samples) for class_samples in data_set_information.test_set.material_samples])
    total_samples_per_class_tot = total_samples_per_class_tr+total_samples_per_class_val+total_samples_per_class_te

    total_pixels_per_class_tr = calculate_total_pixels_per_class(data_set_information.training_set.material_samples)
    total_pixels_per_class_val = calculate_total_pixels_per_class(data_set_information.validation_set.material_samples)
    total_pixels_per_class_te = calculate_total_pixels_per_class(data_set_information.test_set.material_samples)
    total_pixels_per_class_tot = total_samples_per_class_tr + total_pixels_per_class_val + total_pixels_per_class_te

    print 'Total samples per class: {}'.format(list(total_samples_per_class_tot))
    print 'Total pixels per class: {}'.format(list(total_pixels_per_class_tot))
    print 'Total samples: {}'.format(sum(list(total_samples_per_class_tot)))


    print 'Training set'
    print 'Total samples per class: {}'.format(list(total_samples_per_class_tr))
    print 'Total pixels per class: {}'.format(list(total_pixels_per_class_tr))
    print 'Total samples: {}'.format(sum(list(total_samples_per_class_tr)))

    print 'Validation set'
    print 'Total samples per class: {}'.format(list(total_samples_per_class_val))
    print 'Total pixels per class: {}'.format(list(total_pixels_per_class_val))
    print 'Total samples: {}'.format(sum(list(total_samples_per_class_val)))

    print 'Test set'
    print 'Total samples per class: {}'.format(list(total_samples_per_class_te))
    print 'Total pixels per class: {}'.format(list(total_pixels_per_class_te))
    print 'Total samples: {}'.format(sum(list(total_samples_per_class_te)))
