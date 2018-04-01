# coding=utf-8

import json
import os
import time
import argparse
import math

import matplotlib as mpl
import numpy as np
import pydensecrf.utils as dcrf_utils
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from matplotlib import pyplot as plt

from scipy import ndimage

from utils import dataset_utils
from utils import prediction_utils
from utils import image_utils
from PIL import Image

from models import get_model

##############################################
# GLOBALS
##############################################

CONFIG = None
DPI = 120
FIGURE_INDEX = 0


##############################################
# ACTIVATION FUNCTIONS
##############################################

def np_softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if X.ndim == 1:
        p = p.flatten()

    return p


##############################################
# UTILITIES
##############################################

class PredictionImage(object):
    def __init__(self, image_path, data_set_information, sdim=None, scale_factor=1.0, div2_constraint=4, labeled_only=False):
        print 'Loading image from: {}'.format(image_path)
        self.pil_image = load_img(image_path)
        self.file_name = os.path.basename(image_path)

        if sdim is not None:
            print 'Scaling to sdim requirement: {}'.format(sdim)
            self.pil_image = pil_resize_image_to_sdim(self.pil_image, sdim, interp='bicubic')

        print 'Loaded image of size: {}'.format(self.pil_image.size)
        self.np_image = img_to_array(self.pil_image)

        print 'Pre-processing image data with div2 constraint: {} labeled only: {}'.format(div2_constraint, labeled_only)
        per_channel_mean = data_set_information.labeled_per_channel_mean if labeled_only else data_set_information.per_channel_mean
        per_channel_stddev = data_set_information.labeled_per_channel_stddev if labeled_only else data_set_information.per_channel_stddev

        self.original_shape = self.np_image.shape[:2]
        self.scale_factor = scale_factor
        self.scaled = False

        if self.scale_factor != 1.0:
            target_height = int(np.round(self.np_image.shape[0] * self.scale_factor))
            target_width = int(np.round(self.np_image.shape[1] * self.scale_factor))
            target_shape = (target_height, target_width)
            print 'Scaling image with scale factor: {}, from shape: {} to shape: {}'.format(scale_factor, self.original_shape, target_shape)
            self.np_image = image_utils.np_scale_image(self.np_image, sfactor=scale_factor, interp='bicubic')
            self.scaled = True

        self.np_image = image_utils.np_normalize_image_channels(self.np_image, per_channel_mean=per_channel_mean, per_channel_stddev=per_channel_stddev)
        div2_constraint = div2_constraint

        self.padded = False
        self.v_pad_before, self.v_pad_after, self.h_pad_before, self.h_pad_after = 0, 0, 0, 0

        if dataset_utils.count_trailing_zeroes(self.np_image.shape[0]) < div2_constraint or \
                        dataset_utils.count_trailing_zeroes(self.np_image.shape[1]) < div2_constraint:
            self.np_image, self.v_pad_before, self.v_pad_after, self.h_pad_before, self.h_pad_after =\
                pad_image_to_div2_constraint(self.np_image, div2_constraint, data_set_information.per_channel_mean)
            self.padded = True


def pil_resize_image_to_sdim(pil_image, sdim, interp='nearest'):
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    scale_factor = float(sdim) / float(min(pil_image.width, pil_image.height))

    # If no scaling is required
    if scale_factor == 1.0:
        return pil_image

    target_width = int(round(scale_factor * pil_image.width))
    target_height = int(round(scale_factor * pil_image.height))
    resized = pil_image.resize((target_width, target_height), resample=func[interp])
    return resized


def read_config_json(path):
    with open(path) as f:
        data = f.read()
        return json.loads(data)


def get_config_value(key):
    global CONFIG
    return CONFIG[key] if key in CONFIG else None


def get_latest_weights_file_path(weights_folder_path):
    weight_files = dataset_utils.get_files(weights_folder_path)

    if len(weight_files) > 0:
        weight_files.sort()
        weight_file = weight_files[-1]
        return os.path.join(weights_folder_path, weight_file)

    return None


def pad_image_to_div2_constraint(image_array, div2_constraint, cval):
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


def get_new_figure(target_width, target_height, window_title):

    global FIGURE_INDEX

    max_width = 3840.0/DPI
    max_height = 2160.0/DPI
    labelsize = 6

    fig_width = min(target_width, max_width)
    fig_height = min(target_height, max_height)

    # Create figure
    plt.rc('xtick', labelsize=labelsize)
    plt.rc('ytick', labelsize=labelsize)

    figure = plt.figure(FIGURE_INDEX, figsize=(fig_width, fig_height), dpi=DPI)
    figure.canvas.set_window_title(window_title)
    FIGURE_INDEX += 1
    return figure


def build_topk_segmentation_plot(flattened_masks, original_pil_image, file_name, output=None):
    # type: (list[np.ndarray], PIL.Image) -> None

    top_k = len(flattened_masks)
    cbar_shrink = 0.75
    max_images_per_row = 3
    segmentation_map_alpha = 0.65

    # Columns of 3 images (side-by-side)
    cols = min(top_k, max_images_per_row)
    rows = (top_k/max_images_per_row) + 1 if top_k%max_images_per_row != 0 else (top_k/max_images_per_row)
    target_width = (float(original_pil_image.width)/DPI) * cols
    target_height = (float(original_pil_image.height)/DPI) * rows

    figure = get_new_figure(target_width, target_height, 'Top {} segmentation of {}'.format(top_k, file_name))

    # Map the material colors to ones that are easier to recognize for the eye
    material_colors = image_utils.get_distinct_colors(24, seed=1234)

    for i in range(0, top_k):
        flattened_mask = flattened_masks[i][0]
        found_materials = flattened_masks[i][1]
        new_mask = np.array(flattened_mask, copy=True)

        for j in range(0, len(found_materials)):
            old_color = found_materials[j][0].color
            mask = flattened_mask[:,:,0] == old_color[0]
            new_color = np.array(material_colors[found_materials[j][0].id], dtype='float32') * 255.0
            new_mask[mask] = new_color

        flattened_masks[i] = (new_mask, found_materials)

    for i in range(0, top_k):
        flattened_mask = flattened_masks[i][0]
        found_materials = flattened_masks[i][1]
        segmented_img = array_to_img(flattened_mask, scale=False)

        # Add a new subplot
        figure.add_subplot(rows, cols, i+1, title='Top {}'.format(i+1))

        # Create the color map for the color bar
        colors = [np.array(material_colors[m[0].id], dtype='float32') for m in found_materials]
        labels = ['{0} / {1:.2f}%'.format(m[0].name, m[1]) for m in found_materials]
        cmap = mpl.colors.ListedColormap(colors, name='material_colors', N=len(colors))

        # Create the color bar
        num_found_materials = len(found_materials)
        step_size = (1.0 / num_found_materials)
        bounds = np.arange(num_found_materials + 1) * step_size
        ticks = bounds - step_size * 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        if output is not None:
            output_original_img = original_pil_image.copy().convert('RGBA')
            output_segmented_img = segmented_img.copy().convert('RGBA')
            output_segmented_img.putalpha(int(segmentation_map_alpha*255))
            composite_img = Image.alpha_composite(output_original_img, output_segmented_img)
            composite_img.save(output + '_top_{}_segmentation.png'.format(i+1))

        # Create the original image as background and combine with the segmentation mask
        plt_img = plt.imshow(original_pil_image, interpolation='bicubic', origin='upper')
        plt_img = plt.imshow(segmented_img, alpha=segmentation_map_alpha, interpolation='bicubic', origin='upper', cmap=cmap, norm=norm)

        # Set the color bar to the
        cbar = plt.colorbar(plt_img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm, shrink=cbar_shrink)
        cbar.set_ticklabels(labels)

    plt.tight_layout()


def build_topk_accuracy_plot(flattened_predictions, original_pil_image, file_name, ground_truth_mask, output=None):
    # Provide accuracy correct / num unignored pixels

    top_k = len(flattened_predictions)
    cbar_shrink = 0.75
    max_images_per_row = 3
    segmentation_map_alpha = 0.80
    image_height = original_pil_image.height
    image_width = original_pil_image.width

    # Encode values and colors for the different classes: ignored, incorrect and correct pixels
    ignored_val = 0
    incorrect_val = 1
    correct_val = 2

    ignored_color = np.array([0, 0, 0], dtype=np.uint8)
    incorrect_color = np.array([255, 0, 0], dtype=np.uint8)
    correct_color = np.array([0, 255, 0], dtype=np.uint8)

    # Columns of 3 images (side-by-side)
    cols = min(top_k, max_images_per_row)
    rows = (top_k/max_images_per_row) + 1 if top_k%max_images_per_row != 0 else (top_k/max_images_per_row)
    target_width = (float(image_width)/DPI) * cols
    target_height = (float(image_height)/DPI) * rows

    figure = get_new_figure(target_width, target_height, 'Top {} segmentation accuracy of {}'.format(top_k, file_name))
    num_non_ignored_pixels = np.count_nonzero(ground_truth_mask)
    num_total_pixels = image_width*image_height
    ignore_mask = ground_truth_mask == 0

    # Accumulate the top k correct results to this mask
    correct_in_topk_mask = np.zeros(ground_truth_mask.shape, dtype=np.bool)

    # Create a mask with three classes: 0: ignored (bg), 2: incorrect, 1: correct
    for i in range(0, top_k):
        # The material id's are encoded in the red channel of the flattened mask
        flattened_mask = flattened_predictions[i][0][:, :, 0]
        correct_in_mask = flattened_mask == ground_truth_mask

        # Accumulate the results from the i first layers i.e. check whether the correct answer is in the top k guesses
        correct_in_topk_mask = np.logical_or(correct_in_mask, correct_in_topk_mask)

        # Calculate the number of correct and incorrect pixels
        num_correct_pixels = np.count_nonzero(correct_in_topk_mask)
        num_incorrect_pixels = num_non_ignored_pixels - num_correct_pixels
        topk_accuracy = (float(num_correct_pixels)/float(num_non_ignored_pixels))*100.0
        np_vals = np.where(correct_in_topk_mask, correct_val, incorrect_val) * np.invert(ignore_mask).astype(np.int32)
        print 'Top {}: num_correct: {}, num incorrect: {}, num non-ignored: {}, num total pixels: {}'.format(i+1, num_correct_pixels, num_incorrect_pixels, num_non_ignored_pixels, num_total_pixels)

        topk_accuracy_np_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        topk_accuracy_np_img[np_vals == incorrect_val] = incorrect_color
        topk_accuracy_np_img[np_vals == correct_val] = correct_color
        topk_accuracy_np_img[np_vals == ignored_val] = ignored_color

        topk_img = array_to_img(topk_accuracy_np_img)

        if output is not None:
            output_original_img = original_pil_image.copy().convert('RGBA')
            output_top_k_img = topk_img.copy().convert('RGBA')
            output_top_k_img.putalpha(int(segmentation_map_alpha*255))
            composite_img = Image.alpha_composite(output_original_img, output_top_k_img)
            composite_img.save(output + '_top_{}_accuracy.png'.format(i+1))

        # Add a new subplot
        figure.add_subplot(rows, cols, i+1, title='Top {} accuracy {:.2f}%'.format(i+1, topk_accuracy))

        # Create the original image as background and combine with the segmentation mask
        plt_img = plt.imshow(original_pil_image, interpolation='bicubic', origin='upper')
        plt_img = plt.imshow(topk_img, alpha=segmentation_map_alpha, interpolation='bicubic', origin='upper')

    plt.tight_layout()


def save_topk_segmentations(flattened_masks, output_path):

    if output_path is None:
        return

    for i in range(0, len(flattened_masks)):
        flattened_mask = flattened_masks[i][0]
        segmented_img = array_to_img(flattened_mask, scale=False)
        save_file = '{}_top_{}.png'.format(os.path.basename(output_path).split('.')[0], i + 1)
        save_path = os.path.join(os.path.dirname(output_path), save_file)
        print 'Saving top {} predicted segmentation to: {}'.format(i + 1, save_path)
        segmented_img.save(save_path, format='PNG')


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
    ap.add_argument('-t', '--gtruth', required=False, type=str, help="Path to ground truth segmentation mask")
    ap.add_argument('-o', '--output', required=False, type=str, help="Path to output image")
    ap.add_argument('--crf', required=False, type=int, default=0, help="Number of CRF iterations to use")
    ap.add_argument('-k', '--topk', required=False, type=int, default=1, help="Number of top K predictions to show")
    ap.add_argument('--bgweight', required=False, type=float, default=1.0, help="Weight for the background class predictions")
    ap.add_argument('--labeledonly', required=False, type=bool, default=False, help="Was the model trained using labeled only data")
    ap.add_argument('--ensembling', required=False, type=bool, default=False, help="Should we use dimensional ensembling?")
    ap.add_argument('--sdim', required=False, type=int, help='Scale to this sdim before using the image')
    args = vars(ap.parse_args())

    model_name = args['model']
    config_file_path = args['config']
    input_image_path = args['input']
    ground_truth_image_path = args['gtruth']
    weights_path = args['weights']
    output_path = args['output']
    crf_iterations = args['crf']
    top_k = args['topk']
    background_class_prediction_weight = args['bgweight']
    labeled_only = args['labeledonly']
    ensembling = args['ensembling']
    sdim = args['sdim']

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

    # Read the div2 constraint
    div2_constraint = get_config_value('div2_constraint')

    # Load the model
    num_classes = len(material_class_information)
    input_shape = get_config_value('input_shape')

    # If we are using ensembling the network has to be fully convolutional
    if ensembling and (input_shape[0] is not None and input_shape[1] is not None):
        raise ValueError('Cannot use dimensional ensembling if the input shape is not variable')

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
    print 'Using ensembling: {}'.format(ensembling)

    # Create the prediction images
    prediction_images = []

    # If using ensembling create the different sized versions
    if ensembling:
        # TODO: Figure out whether scale factors or or set sdim sizes are better
        ENSEMBLING_SCALE_FACTORS = [1.0/math.sqrt(2.0), 1.0, math.sqrt(2)] # [0.50, 0.75, 1.0] # [1.0, 1.5, 2.0] #
        print 'Using ensembling scale factors: {}'.format(ENSEMBLING_SCALE_FACTORS)

        for sfactor in ENSEMBLING_SCALE_FACTORS:
            prediction_images.append(
                PredictionImage(input_image_path, data_set_information, sdim=sdim, scale_factor=sfactor, div2_constraint=div2_constraint, labeled_only=labeled_only))
    # If not using ensembling we are using only scale factor of 1.0
    else:
        prediction_images.append(
            PredictionImage(input_image_path, data_set_information, sdim=sdim, scale_factor=1.0, div2_constraint=div2_constraint, labeled_only=labeled_only))

    predictions = []

    for prediction_image in prediction_images:
        print 'Predicting segmentation for image with shape: {}'.format(prediction_image.np_image.shape)

        # The model is expecting a batch size, even if it's one so append
        # one new dimension to the beginning to mark batch size of one - squeeze when done
        start_time = time.time()
        prediction = model.predict(prediction_image.np_image[np.newaxis, :])
        prediction = prediction.squeeze()
        predictions.append(prediction)
        end_time = time.time()

        print 'Prediction finished in time: {} s'.format(end_time - start_time)

    # Undo the transformations to the images: scaling and padding
    # And apply background class prediction weight
    for i, prediction_image in enumerate(prediction_images):
        # First remove the padding from possible div2 constraint
        if prediction_image.padded:
            print 'Cropping padding regions from predictions'
            predictions[i] = image_utils.np_crop_image(predictions[i],
                                        prediction_image.h_pad_before,
                                        prediction_image.v_pad_before,
                                        prediction_image.np_image.shape[1] - prediction_image.h_pad_after,
                                        prediction_image.np_image.shape[0] - prediction_image.v_pad_after)

            print 'Shape after cropping: {}'.format(predictions[i].shape)

        # Scale back to original size
        if prediction_image.scaled:
            sfactor = 1.0 / prediction_image.scale_factor
            print 'Scaling predictions back to original image shape: {} with scale factor: {}'.format(prediction_image.original_shape, sfactor)
            scaled_prediction = ndimage.zoom(predictions[i],
                                             zoom=(sfactor, sfactor, 1.0),
                                             order=image_utils.ImageInterpolationType.BICUBIC.value,
                                             mode='constant',
                                             cval=-1.0).astype(predictions[i].dtype)

            # TODO: Fix these off by one issues due to rounding
            if scaled_prediction.shape[0] > prediction_image.original_shape[0]:
                hdiff = scaled_prediction.shape[0] - prediction_image.original_shape[0]
                scaled_prediction = scaled_prediction[hdiff:, :, :]

            if scaled_prediction.shape[1] > prediction_image.original_shape[1]:
                wdiff = scaled_prediction.shape[1] - prediction_image.original_shape[1]
                scaled_prediction = scaled_prediction[:, wdiff:, :]

            if scaled_prediction.shape[0] < prediction_image.original_shape[0] or scaled_prediction.shape[1] < prediction_image.original_shape[1]:
                scaled_prediction = image_utils.np_pad_image_to_shape(scaled_prediction, prediction_image.original_shape, 0.0)

            predictions[i] = scaled_prediction

        if predictions[i].shape[:2] != prediction_image.original_shape:
            raise ValueError('Image shape after undoing transformations does not match the original shape: {} vs {}'
                             .format(predictions[i].shape[:2], prediction_image.original_shape))

        # Weight the background class activations to reduce the salt'n'pepper noise
        # likely caused by the class imbalance in the training data
        predictions[i][:, :, 0] *= background_class_prediction_weight

    # Take the mean of all the predictions as the final prediction
    final_prediction = np.mean(np.array(predictions), axis=0)

    original_pil_image = prediction_images[-1].pil_image
    file_name = prediction_images[-1].file_name

    # Run CRF if we are using it for post-processing
    # TODO: Before or after voting (averaging)? The padding and would at least have to be removed before - a lot more expensive to run for each
    # any CRF is run
    if crf_iterations > 0:
        # Must use the unnormalized image data
        original_image_array = img_to_array(original_pil_image)
        crf = prediction_utils.get_dcrf(original_image_array, num_classes)

        # Turn the output of the last convolutional layer to softmax probabilities
        # and then to unary.
        softmax = np_softmax(final_prediction, axis=-1)
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
        final_prediction = np.reshape(Q, np.flip(final_prediction.shape, 0), order='A')
        final_prediction = np.transpose(final_prediction, (2, 1, 0))

    flattened_predictions = prediction_utils.top_k_flattened_masks(final_prediction, top_k, material_class_information, True)

    if ground_truth_image_path:
        print 'Reading ground truth image from: {}'.format(ground_truth_image_path)
        ground_truth_pil_image = load_img(ground_truth_image_path)

        if sdim is not None:
            ground_truth_pil_image = pil_resize_image_to_sdim(ground_truth_pil_image, sdim=sdim, interp='nearest')

        ground_truth_np_img = img_to_array(ground_truth_pil_image)
        ground_truth_mask = dataset_utils.index_encode_mask(np_mask_img=ground_truth_np_img, material_class_information=material_class_information)
        print 'Building top k segmentation accuracy plot'
        build_topk_accuracy_plot(flattened_predictions, original_pil_image, file_name, ground_truth_mask, output=output_path)

    print 'Building top k segmentation plot'
    build_topk_segmentation_plot(flattened_predictions, original_pil_image, file_name, output=output_path)

    # Show all the plots
    plt.show()

    #if output_path is not None:
    #    save_topk_segmentations(flattened_predictions, output_path)


if __name__ == '__main__':
    main()
