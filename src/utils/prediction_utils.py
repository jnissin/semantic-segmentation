# coding=utf-8

import numpy as np
from dataset_utils import MaterialClassInformation
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary


def predictions_to_image(predictions, material_class_information, verbose=False):
    # type: (np.array, list[MaterialClassInformation], bool) -> (np.array, list[(str, float)])

    """
    Flattens the segmentation predictions to an image from the HxWx1 form where each
    entry reflects the material index. Returns the image and a list of the found material
    names and percentages.

    # Arguments
        :param predictions: the top predictions as a HxWx3 numpy array (image)
        :param material_class_information: the material class information to map the predictions to right classes
        :param verbose: should the function print information about the run

    # Returns
        :return: the predictions as a flattened image and a list of found materials and their percentages
    """
    flattened_mask = np.zeros(shape=(predictions.shape[0], predictions.shape[1], 3), dtype='uint8')
    found_materials = []
    image_pixels = float(predictions.shape[0] * predictions.shape[1])

    for material_class in material_class_information:
        material_class_id = material_class.id

        # Select all the pixels with the corresponding id values
        class_mask = predictions[:, :] == material_class_id

        # Set all the corresponding pixels in the flattened image
        # to the material color. If there are many colors for one
        # material, select the first to represent them all.
        # material_r_color = material_class.r_color_values[0]

        # Parse a unique color for the material
        color = material_class.color

        if verbose and np.any(class_mask):
            percentage = (float(np.sum(class_mask)) / image_pixels) * 100.0
            print 'Found material: {}, in {}% of the pixels. Assigning it color: {}' \
                .format(material_class.name, percentage, color)
            found_materials.append((material_class, percentage))

            # Assign the material color to all the masked pixels
            flattened_mask[class_mask] = color

    if verbose:
        print 'Found in total {} materials'.format(len(found_materials))

    return flattened_mask, found_materials


def flatten_mask(expanded_mask, material_class_information, verbose=False):
    # type: (np.array, list[MaterialClassInformation], bool) -> (np.array, list[(str, float)])
    """
    Flattens the prediction from the expanded HxWxNUM_CLASSES form to an image. Uses
    argmax to select the best predictions and flattens that to an image.

    # Arguments
        :param expanded_mask: the predictions in expanded HxWxNUM_CLASSES form
        :param material_class_information:
        :param verbose: should the function print information about it's run
    # Returns
        :return: the predictions as a flattened image and a list of found materials and their percentages
    """
    # The predictions now reflect material class ids
    predictions = np.argmax(expanded_mask, axis=-1)
    return predictions_to_image(predictions, material_class_information, verbose)


def top_k_flattened_masks(expanded_mask, k, material_class_information, verbose=False):
    # type: (np.array, int, list[MaterialClassInformation], bool) -> list[(np.array, list[(str, float)])]

    """
    Returns k different predictions describing the top k predictions for each pixel.

    # Arguments
        :param expanded_mask: the predictions in expanded HxWxNUM_CLASSES form
        :param k: how many predictions to return
        :param material_class_information:
        :param verbose: should the function print information about it's run
    # Returns
        :return: a list of top k predictions as a flattened images and a lists of found materials and their percentages
    """

    flattened_masks = []

    # Get the top k predictions
    top_k_predictions = np.argsort(-expanded_mask)
    top_k_predictions = np.transpose(top_k_predictions)
    top_k_predictions = top_k_predictions[:k]
    top_k_predictions = np.transpose(top_k_predictions, axes=(0, 2, 1))

    for i in range(0, k):
        print 'Processing top {} segmentation'.format(i + 1)
        flattened_masks.append(predictions_to_image(top_k_predictions[i], material_class_information, verbose))

    return flattened_masks


def get_dcrf(img, nlabels):
    width = img.shape[1]
    height = img.shape[0]
    img_shape = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, nlabels)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(5, 5), shape=img_shape)

    d.addPairwiseEnergy(feats,
                        compat=4,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80),
                                      schan=(20, 20, 20),
                                      img=img,
                                      chdim=2)

    d.addPairwiseEnergy(feats,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    return d
