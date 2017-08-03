# coding=utf-8

import json
import os
import sys
import time
import argparse

import matplotlib as mpl
import numpy as np
import pydensecrf.utils as dcrf_utils
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from matplotlib import pyplot as plt

from utils import dataset_utils
from utils import prediction_utils
from utils import image_utils

import losses
from models.models import get_model

##############################################
# GLOBALS
##############################################

CONFIG = None


##############################################
# UTILITIES
##############################################

class PredictionImage(object):
    def __init__(self, image_path, data_set_information):
        print 'Loading image from: {}'.format(image_path)
        self.pil_image = load_img(image_path)

        print 'Loaded image of size: {}'.format(self.pil_image.size)
        self.np_image = img_to_array(self.pil_image)

        print 'Normalizing image data'
        self.np_image = image_utils.np_normalize_image_channels(self.np_image, data_set_information.per_channel_mean, data_set_information.per_channel_stddev)

        div2_constraint = 4

        self.padded = False
        self.v_pad_before, self.v_pad_after, self.h_pad_before, self.h_pad_after = 0, 0, 0, 0

        if dataset_utils.count_trailing_zeroes(self.np_image.shape[0]) < div2_constraint or \
                        dataset_utils.count_trailing_zeroes(self.np_image.shape[1]) < div2_constraint:
            self.np_image, self.v_pad_before, self.v_pad_after, self.h_pad_before, self.h_pad_after =\
                pad_image(self.np_image, div2_constraint, data_set_information.per_channel_mean)
            self.padded = True


def read_config_json(path):
    with open(path) as f:
        data = f.read()
        return json.loads(data)


def get_config_value(key):
    global CONFIG
    return CONFIG[key] if key in CONFIG else None


def set_config_value(key, value):
    global CONFIG
    CONFIG[key] = value


def get_latest_weights_file_path(weights_folder_path):
    weight_files = dataset_utils.get_files(weights_folder_path)

    if len(weight_files) > 0:
        weight_files.sort()
        weight_file = weight_files[-1]
        return os.path.join(weights_folder_path, weight_file)

    return None


def pad_image(image_array, div2_constraint, cval):
    padded_height = dataset_utils.get_closest_higher_number_with_n_trailing_zeroes(image_array.shape[0], div2_constraint)
    padded_width = dataset_utils.get_closest_higher_number_with_n_trailing_zeroes(image_array.shape[1], div2_constraint)
    padded_shape = (padded_height, padded_width)

    print 'Padding image from {} to {}'.format(image_array.shape, padded_shape)

    v_diff = max(0, padded_shape[0] - image_array.shape[0])
    h_diff = max(0, padded_shape[1] - image_array.shape[1])

    v_pad_before = v_diff / 2
    v_pad_after = (v_diff / 2) + (v_diff % 2)

    h_pad_before = h_diff / 2
    h_pad_after = (h_diff / 2) + (h_diff % 2)

    padded_image_array = image_utils.np_pad_image(
        image_array,
        v_pad_before,
        v_pad_after,
        h_pad_before,
        h_pad_after,
        cval)

    return padded_image_array, v_pad_before, v_pad_after, h_pad_before, h_pad_after


def show_segmentation_plot(figure_ind,
                           title,
                           segmented_img,
                           found_materials,
                           original_image=None):

    colors = [np.array(m[0].color, dtype='float32')/255.0 for m in found_materials]
    labels = ['{0} / {1:.4f}%'.format(m[0].name, m[1]) for m in found_materials]
    cmap = mpl.colors.ListedColormap(colors, name='material_colors', N=len(colors))

    # Create figure
    fig_width = max(float(segmented_img.width)/100.0, 8.0)
    fig_height = max(float(segmented_img.height)/100.0, 8.0)
    f = plt.figure(figure_ind, figsize=(fig_width, fig_height), dpi=100)
    f.suptitle(title)

    # Create the original image as background
    if original_image:
        orig_img = plt.imshow(original_image, interpolation='bilinear', origin='upper')

    # Create the color bar
    height = float(segmented_img.size[1])
    num_materials = len(found_materials)
    step = (float(height) / float(num_materials))
    bounds = np.arange(num_materials+1) * step
    ticks = bounds - step*0.5
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Create the plt img
    alpha = 1.0 if not original_image else 0.65
    plt_img = plt.imshow(segmented_img, alpha=alpha, interpolation='bilinear', origin='upper', cmap=cmap, norm=norm)

    # Create the color bar
    cbar = plt.colorbar(plt_img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm)
    cbar.set_ticklabels(labels)

    # Show the segmentation
    f.show()


##############################################
# MAIN
##############################################

def main():

    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation.')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-w', '--weights', required=True, type=str, help="Path to weights directory or weights file")
    ap.add_argument('-i', '--input', required=True, type=str, help="Path to input image")
    ap.add_argument('-o', '--output', required=False, type=str, help="Path to output image")
    ap.add_argument('--crf', required=False, type=int, default=0, help="Number of CRF iterations to use")
    ap.add_argument('-k', '--topk', required=False, type=int, default=1, help="Number of top K predictions to show")
    ap.add_argument('--bgweight', required=False, type=float, default=1.0, help="Weight for the background class predictions")
    args = vars(ap.parse_args())

    model_name = args['model']
    config_file_path = args['config']
    input_image_path = args['input']
    weights_path = args['weights']
    output_path = args['output']
    crf_iterations = args['crf']
    top_k = args['topk']
    background_class_prediction_weight = args['bgweight']

    # Read the configuration file
    global CONFIG
    print 'Loading the configuration from file: {}'.format(config_file_path)
    CONFIG = read_config_json(config_file_path)

    # Load the material class information
    material_class_information_path = get_config_value('path_to_material_class_file')
    print 'Loading the material class information from file: {}'.format(material_class_information_path)
    material_class_information = dataset_utils.load_material_class_information(material_class_information_path)
    print 'Loaded {} material classes'.format(len(material_class_information))

    # Load the data set information
    data_set_information_path = get_config_value('path_to_data_set_information_file')
    print 'Loading data set information from file: {}'.format(data_set_information_path)
    data_set_information = dataset_utils.load_segmentation_data_set_information(data_set_information_path)
    print 'Data set information loaded'

    # Load the model
    num_classes = len(material_class_information)
    input_shape = get_config_value('input_shape')

    print 'Loading model {} instance with input shape: {}, num classes: {}'.format(model_name, input_shape, num_classes)
    model_wrapper = get_model(model_name, input_shape, num_classes)
    model = model_wrapper.model

    # Load either provided weights or try to find the newest weights from the
    # checkpoint path
    if os.path.isdir(weights_path):
        print 'Searching for most recent weights in: {}'.format(weights_path)
        weights_path = get_latest_weights_file_path(weights_path)

    print 'Loading weights from: {}'.format(weights_path)
    model.load_weights(weights_path)

    # Load the image file
    prediction_image = PredictionImage(input_image_path, data_set_information)

    # The model is expecting a batch size, even if it's one so append
    # one new dimension to the beginning to mark batch size of one
    print 'Predicting segmentation for image with shape: {}'.format(prediction_image.np_image.shape)
    start_time = time.time()
    expanded_mask = model.predict(prediction_image.np_image[np.newaxis, :])
    end_time = time.time()
    print 'Prediction finished in time: {} s'.format(end_time - start_time)

    # Select the only image from the batch i.e. remove single-dimensional
    # entries from the shape array
    expanded_mask = expanded_mask.squeeze()

    if prediction_image.padded:
        print 'Cropping predictions back to original image shape'
        expanded_mask = image_utils.np_crop_image(expanded_mask,
                                    prediction_image.h_pad_before,
                                    prediction_image.v_pad_before,
                                    prediction_image.np_image.shape[1] - prediction_image.h_pad_after,
                                    prediction_image.np_image.shape[0] - prediction_image.v_pad_after)

        print 'Size after cropping: {}'.format(expanded_mask.shape)

    # Weight the background class activations to reduce the salt'n'pepper noise
    # likely caused by the class imbalance in the training data
    expanded_mask[:, :, 0] = expanded_mask[:, :, 0] * background_class_prediction_weight

    # Run CRF if we are using it for post-processing
    if crf_iterations > 0:
        # Must use the unnormalized image data
        original_image_array = img_to_array(prediction_image.pil_image)
        crf = prediction_utils.get_dcrf(original_image_array, num_classes)

        # Turn the output of the last convolutional layer to softmax probabilities
        # and then to unary.
        softmax = losses.np_softmax(expanded_mask, axis=-1)
        softmax = softmax.transpose((2, 0, 1))
        unary = dcrf_utils.unary_from_softmax(softmax)

        # Set the unary; The inputs should be C-continious since
        # we are using Cython wrapper
        unary = np.ascontiguousarray(unary)
        crf.setUnaryEnergy(unary)

        # Run the CRF
        print 'Running CRF for {} iterations'.format(crf_iterations)
        crf_start_time = time.time()
        Q = crf.inference(crf_iterations)
        crf_end_time = time.time()
        print 'CRF inference finished in time: {} s'.format(crf_end_time - crf_start_time)

        # Reshape the outcome
        expanded_mask = np.reshape(Q, np.flip(expanded_mask.shape, 0), order='A')
        expanded_mask = np.transpose(expanded_mask, (2, 1, 0))

    flattened_masks = prediction_utils.top_k_flattened_masks(expanded_mask, top_k, material_class_information, True)

    for i in range(0, top_k):
        flattened_mask = flattened_masks[i][0]
        found_materials = flattened_masks[i][1]
        segmented_img = array_to_img(flattened_mask, scale=False)
        title = 'Top {} segmentation of {}'.format(i+1, os.path.basename(input_image_path))
        show_segmentation_plot(i+1, title, segmented_img, found_materials, prediction_image.pil_image)

        if output_path is not None:
            save_file = '{}_top_{}.png'.format(os.path.basename(output_path).split('.')[0], i+1)
            save_path = os.path.join(os.path.dirname(output_path), save_file)
            print 'Saving top {} predicted segmentation to: {}'.format(i+1, save_path)
            segmented_img.save(save_path, format='PNG')

    # Keep figures alive until user input from console
    raw_input()
    print 'Done'


if __name__ == '__main__':
    main()