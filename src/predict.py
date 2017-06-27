# coding=utf-8

import json
import os
import sys
import time

import matplotlib as mpl
import numpy as np
import pydensecrf.utils as dcrf_utils
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from matplotlib import pyplot as plt

from models import model_utils
from utils import dataset_utils

##############################################
# GLOBALS
##############################################

CONFIG = None


##############################################
# UTILITIES
##############################################

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
    padded_height = dataset_utils.get_closest_number_with_n_trailing_zeroes(image_array.shape[0], div2_constraint)
    padded_width = dataset_utils.get_closest_number_with_n_trailing_zeroes(image_array.shape[1], div2_constraint)
    padded_shape = (padded_height, padded_width)

    print 'Padding image from {} to {}'.format(image_array.shape, padded_shape)

    v_diff = max(0, padded_shape[0] - image_array.shape[0])
    h_diff = max(0, padded_shape[1] - image_array.shape[1])

    v_pad_before = v_diff / 2
    v_pad_after = (v_diff / 2) + (v_diff % 2)

    h_pad_before = h_diff / 2
    h_pad_after = (h_diff / 2) + (h_diff % 2)

    padded_image_array = dataset_utils.np_pad_image(
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

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Invalid number of arguments, usage: ./{} <path_to_config> <path_to_image> <opt: saved_mask_name> <opt: path_to_weights>'.format(
            sys.argv[0])
        sys.exit(0)

    # Read the configuration file
    print 'Loading the configuration from file: {}'.format(sys.argv[1])
    CONFIG = read_config_json(sys.argv[1])

    # Load the material class information
    material_class_information_path = get_config_value('path_to_material_class_file')
    print 'Loading the material class information from file: {}'.format(material_class_information_path)
    material_class_information = dataset_utils.load_material_class_information(material_class_information_path)
    print 'Loaded {} material classes'.format(len(material_class_information))

    # Load the model
    model_name = get_config_value('model')
    num_classes = len(material_class_information)
    num_channels = get_config_value('num_channels')
    input_shape = (None, None, num_channels)

    print 'Loading model {} instance with input shape: {}, num classes: {}'\
        .format(model_name, input_shape, num_classes)

    model = model_utils.get_model(model_name, input_shape, num_classes)

    # Load either provided weights or try to find the newest weights from the
    # checkpoint path
    weights_file_path = None

    if len(sys.argv) > 4:
        weights_file_path = sys.argv[4]

    if not weights_file_path:
        weights_directory_path = os.path.dirname(get_config_value('keras_model_checkpoint_file_path'))
        print 'Searching for most recent weights in: {}'.format(weights_directory_path)
        weights_file_path = get_latest_weights_file_path(weights_directory_path)

    if not weights_file_path:
        print 'No existing weights found, exiting'
        sys.exit(0)

    print 'Loading weights from: {}'.format(weights_file_path)
    model.load_weights(weights_file_path)

    # Load the image
    image_path = sys.argv[2]

    print 'Loading image from: {}'.format(image_path)
    image = load_img(image_path)

    print 'Loaded image of size: {}'.format(image.size)
    image_array = img_to_array(image)

    print 'Normalizing image data'
    image_array = dataset_utils.normalize_image_channels(
        image_array,
        np.array(get_config_value('per_channel_mean')),
        np.array(get_config_value('per_channel_stddev')))

    div2_constraint = 4
    print 'Checking image size constraints with div 2 constraint: {}'.format(div2_constraint)

    v_pad_before, v_pad_after, h_pad_before, h_pad_after = 0, 0, 0, 0
    padded = False

    if dataset_utils.count_trailing_zeroes(image_array.shape[0]) < div2_constraint or \
                    dataset_utils.count_trailing_zeroes(image_array.shape[1]) < div2_constraint:

        image_array, v_pad_before, v_pad_after, h_pad_before, h_pad_after \
            = pad_image(image_array, div2_constraint, get_config_value('per_channel_mean'))
        padded = True

    # The model is expecting a batch size, even if it's one so append
    # one new dimension to the beginning to mark batch size of one
    print 'Predicting segmentation for {} size image'.format(image.size)
    start_time = time.time()
    expanded_mask = model.predict(image_array[np.newaxis, :])
    end_time = time.time()
    print 'Prediction finished in time: {} s'.format(end_time - start_time)

    # Select the only image from the batch i.e. remove single-dimensional
    # entries from the shape array
    expanded_mask = expanded_mask.squeeze()

    if padded:
        print 'Cropping predictions back to original image shape'
        expanded_mask = dataset_utils.np_crop_image(expanded_mask,
                                    h_pad_before,
                                    v_pad_before,
                                    image_array.shape[1] - h_pad_after,
                                    image_array.shape[0] - v_pad_after)

        print 'Size after cropping: {}'.format(expanded_mask.shape)

    # Weight the background class activations to reduce the salt'n'pepper noise
    # likely caused by the class imbalance in the training data
    background_class_prediction_weight = get_config_value('background_class_prediction_weight')

    if background_class_prediction_weight is not None:
        expanded_mask[:, :, 0] = expanded_mask[:, :, 0] * background_class_prediction_weight

    # Check whether we are using a CRF
    if get_config_value('use_crf_in_prediction'):
        # Must use the unnormalized image data
        original_image_array = img_to_array(image)
        crf_iterations = get_config_value('crf_iterations')
        crf = model_utils.get_dcrf(original_image_array, num_classes)

        # Turn the output of the last convolutional layer to softmax probabilities
        # and then to unary.
        softmax = model_utils.np_softmax(expanded_mask, axis=-1)
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

    k = 3
    flattened_masks = dataset_utils.top_k_flattened_masks(expanded_mask, k, material_class_information, True)

    for i in range(0, k):
        flattened_mask = flattened_masks[i][0]
        found_materials = flattened_masks[i][1]
        segmented_img = array_to_img(flattened_mask, scale=False)
        title = 'Top {} segmentation of {}'.format(i+1, image_path.split('/')[-1])
        show_segmentation_plot(i+1, title, segmented_img, found_materials, image)

        if len(sys.argv) > 3:
            save_file = 'top_{}_{}'.format(i+1, sys.argv[3])
            print 'Saving top {} predicted segmentation to: {}'.format(i+1, save_file)
            segmented_img.save(save_file)

    # Keep figures alive until user input from console
    raw_input()
    print 'Done'
