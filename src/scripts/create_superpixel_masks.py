# coding=utf-8

import argparse
import multiprocessing
import os
import numpy as np

from PIL import ImageFile
from joblib import Parallel, delayed

from src.enums import SuperpixelSegmentationFunctionType
from src.utils import image_utils, general_utils
from skimage import exposure


_NUM_PROCESSED_IMAGES = multiprocessing.Value('i', 0)


def generate_mask_for_unlabeled_image(f_type, img_path, output_path, verbose=False, ignore_existing=False, borders_only=False, border_connectivity=2, equalize=False, dtype='float'):
    # type: (SuperpixelSegmentationFunctionType, str, str, bool) -> ()
    global _NUM_PROCESSED_IMAGES

    pil_img = image_utils.load_img(img_path)
    np_img = image_utils.img_to_array(pil_img)

    # Equalize and ensure the image is in the desired format
    if equalize:
        np_img = np_img.astype(dtype=np.uint8)
        np_img = exposure.equalize_adapthist(np_img, nbins=256)

        if dtype == 'float':
            np_img -= 0.5
            np_img *= 2.0
        elif dtype == 'uint8':
            np_img *= 255
            np_img = np_img.astype(np.uint8)
        else:
            raise ValueError('Invalid dtype parameter: {}'.format(dtype))
    else:
        if dtype == 'float':
            np_img = np_img.astype(np.float64)
            np_img /= 255.0
            np_img -= 0.5
            np_img *= 2.0
        elif dtype == 'uint8':
            np_img = np_img.astype(dtype=np.uint8)
        else:
            raise ValueError('Invalid dtype parameter: {}'.format(dtype))

    # Check that the output path exists and is valid
    if os.path.isdir(output_path):
        mask_save_path = os.path.join(output_path, os.path.basename(img_path))
    else:
        mask_save_path = output_path

    # Check that the output format is PNG if not replace it to be. JPG compression
    # can cause issues with masks
    mask_save_path = os.path.splitext(mask_save_path)[0] + '.png'

    # Check if existing should be ignored
    if ignore_existing and os.path.exists(mask_save_path):
        if verbose:
            _NUM_PROCESSED_IMAGES.value += 1
            print 'Num processed masks: {} - mask already exists: {}'.format(_NUM_PROCESSED_IMAGES.value, mask_save_path)
        return

    # Segment
    if f_type == SuperpixelSegmentationFunctionType.FELZENSZWALB:
        mask = image_utils.np_get_felzenszwalb_segmentation(np_img, scale=1000, sigma=0.8, min_size=250, normalize_img=False, borders_only=borders_only, border_connectivity=border_connectivity)
    elif f_type == SuperpixelSegmentationFunctionType.SLIC:
        mask = image_utils.np_get_slic_segmentation(np_img, n_segments=400, sigma=1, compactness=10.0, max_iter=20, normalize_img=False, borders_only=borders_only, border_connectivity=border_connectivity)
    elif f_type == SuperpixelSegmentationFunctionType.QUICKSHIFT:
        mask = image_utils.np_get_quickshift_segmentation(np_img, kernel_size=10, max_dist=20, ratio=0.5, sigma=0.0, normalize_img=False, borders_only=borders_only, border_connectivity=border_connectivity)
    elif f_type == SuperpixelSegmentationFunctionType.WATERSHED:
        mask = image_utils.np_get_watershed_segmentation(np_img, markers=400, compactness=0.0, normalize_img=False, borders_only=borders_only, border_connectivity=border_connectivity)
    else:
        raise ValueError('Unknown label generation function type: {}'.format(f_type))

    # Print information
    if verbose:
        _NUM_PROCESSED_IMAGES.value += 1
        print 'Num processed masks: {} - saving mask to: {}'.format(_NUM_PROCESSED_IMAGES.value, mask_save_path)

    # Generate a colored segmentation image with N distinct borders
    if not borders_only:
        new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float64)

        unique_values = np.unique(mask)
        unique_colors = image_utils.get_distinct_colors(len(unique_values), seed=42)

        for i, val in enumerate(unique_values):
            new_mask[mask == val] = unique_colors[i]

        new_mask *= 255.0
        new_mask = new_mask.astype(dtype=np.uint8)
        mask = new_mask
    # Create borders with black and everything else in white
    else:
        mask = np.expand_dims(mask, -1)

        # Map all values from [0, 1] to [0, 255]
        mask = mask.astype(dtype=np.uint8)
        mask *= 255

    img = image_utils.array_to_img(mask, scale=False)
    img.save(mask_save_path)


def main():

    ap = argparse.ArgumentParser(description="Creates a new filtered set of all the images where the proportion of background pixels is less than specified")
    ap.add_argument("-p", "--photos", required=True, type=str, help="Path to photos folder")
    ap.add_argument("-o", "--output", required=True, type=str, help="Path to the output folder")
    ap.add_argument("-f", "--function", required=True, type=str, choices=['slic', 'felzenswalb', 'quickshift', 'watershed'], help="Segmentation function to use")
    ap.add_argument("-j", "--jobs", required=False, default=8, type=int, help="Number of parallel processes")
    ap.add_argument("-v", "--verbose", required=False, default=True, type=bool, help="Verbose mode")
    ap.add_argument("--dtype", required=False, default='float', choices=['float', 'uint8'])
    ap.add_argument("--equalize", required=False, default=False, type=bool, help="Use histogram equalization")
    ap.add_argument("--iexisting", required=False, default=False, type=bool, help="Do not replace existing masks")
    ap.add_argument("--bonly", required=False, default=False, type=bool, help="Borders only segmentation")
    ap.add_argument("--bconnectivity", required=False, default=2, type=int, help="Border connectivity for border only segmentation (1 or 2)")
    args = vars(ap.parse_args())

    function_name_to_type = {'slic': SuperpixelSegmentationFunctionType.SLIC,
                             'felzenswalb': SuperpixelSegmentationFunctionType.FELZENSZWALB,
                             'quickshift': SuperpixelSegmentationFunctionType.QUICKSHIFT,
                             'watershed': SuperpixelSegmentationFunctionType.WATERSHED}

    photos_path = args['photos']
    output_path = args['output']
    function_name = args['function']
    function_type = function_name_to_type.get(function_name, SuperpixelSegmentationFunctionType.NONE)
    n_jobs = args['jobs']
    verbose = args['verbose']
    ignore_existing = args['iexisting']
    borders_only = args['bonly']
    bconnectivity = args['bconnectivity']
    equalize = args['equalize']
    dtype = args['dtype']

    print 'Input directory: {}'.format(photos_path)
    print 'Output directory: {}'.format(output_path)
    print 'Using {} parallel jobs'.format(n_jobs)
    print 'Using segmentation function: {}'.format(function_name)
    print 'Ignoring existing masks: {}'.format(ignore_existing)
    print 'Using adaptive histogram equalization: {}'.format(equalize)
    print 'Using borders only: {}'.format(borders_only)
    print 'Using border connectivity: {}'.format(bconnectivity)
    print 'Using data type: {} for segmentation'.format(dtype)

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if os.path.isdir(photos_path):
        print 'Reading photos from: {}'.format(photos_path)
        photos = image_utils.list_pictures(photos_path)
    else:
        photos = [photos_path]

    print 'Found {} photos'.format(len(photos))

    print 'Ensuring the output path: {} exists'.format(output_path)
    general_utils.create_path_if_not_existing(output_path)

    if ignore_existing:
        existing_masks = image_utils.list_pictures(output_path)
        print 'Found {} existing masks'.format(len(existing_masks))

        # Remove file extension from mask paths
        mask_file_names_without_ext = set([os.path.basename(m.split('.'))[0] for m in existing_masks])

        # Filter out photos that match the existing mask file names
        photos_without_mask = [p for p in photos if os.path.basename(p).split('.')[0] not in mask_file_names_without_ext]
        print 'Ignoring {} matching photos'.format(len(photos) - len(photos_without_mask))
        photos = photos_without_mask

    print 'Starting to create image masks with function: {}'.format(function_name)
    Parallel(n_jobs=n_jobs, backend='multiprocessing')\
        (delayed(generate_mask_for_unlabeled_image)(function_type, img_path, output_path, verbose, ignore_existing, borders_only, bconnectivity, equalize, dtype) for img_path in photos)

    print 'Done'


if __name__ == '__main__':
    main()
