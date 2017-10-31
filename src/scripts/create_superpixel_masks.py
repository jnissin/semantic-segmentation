# coding=utf-8

import argparse
import multiprocessing
import os
import numpy as np

from PIL import ImageFile
from joblib import Parallel, delayed

from src.enums import SuperpixelSegmentationFunctionType
from src.utils import image_utils, general_utils


_NUM_PROCESSED_IMAGES = multiprocessing.Value('i', 0)


def generate_mask_for_unlabeled_image(f_type, img_path, output_path, verbose=False):
    # type: (SuperpixelSegmentationFunctionType, str, str, bool) -> ()
    global _NUM_PROCESSED_IMAGES

    np_img = image_utils.img_to_array(image_utils.load_img(img_path))
    mask_save_path = os.path.join(output_path, os.path.basename(img_path))

    # Skip existing
    if os.path.exists(mask_save_path):
        if verbose:
            _NUM_PROCESSED_IMAGES.value += 1
            print 'Num processed masks: {} - mask already exists: {}'.format(_NUM_PROCESSED_IMAGES.value, mask_save_path)
        return

    if f_type == SuperpixelSegmentationFunctionType.FELZENSWALB:
        mask = image_utils.np_get_felzenswalb_segmentation(np_img, scale=700, sigma=0.6, min_size=250, normalize_img=True, borders_only=True)
    elif f_type == SuperpixelSegmentationFunctionType.SLIC:
        mask = image_utils.np_get_slic_segmentation(np_img, n_segments=300, sigma=1, compactness=10.0, max_iter=20, normalize_img=True, borders_only=True)
    elif f_type == SuperpixelSegmentationFunctionType.QUICKSHIFT:
        mask = image_utils.np_get_quickshift_segmentation(np_img, kernel_size=20, max_dist=15, ratio=0.5, normalize_img=True, borders_only=True)
    elif f_type == SuperpixelSegmentationFunctionType.WATERSHED:
        mask = image_utils.np_get_watershed_segmentation(np_img, markers=250, compactness=0.001, normalize_img=True, borders_only=True)
    else:
        raise ValueError('Unknown label generation function type: {}'.format(f_type))

    if verbose:
        _NUM_PROCESSED_IMAGES.value += 1
        print 'Num processed masks: {} - saving mask to: {}'.format(_NUM_PROCESSED_IMAGES.value, mask_save_path)

    mask = np.expand_dims(mask, -1)*255
    img = image_utils.array_to_img(mask, scale=False)
    img.save(mask_save_path)


def main():

    ap = argparse.ArgumentParser(description="Creates a new filtered set of all the images where the proportion of background pixels is less than specified")
    ap.add_argument("-p", "--photos", required=True, type=str, help="Path to photos folder")
    ap.add_argument("-o", "--output", required=True, type=str, help="Path to the output folder")
    ap.add_argument("-f", "--function", required=True, type=str, choices=['slic', 'felzenswalb', 'quickshift', 'watershed'], help="Segmentation function to use")
    ap.add_argument("-j", "--jobs", required=False, default=8, type=int, help="Number of parallel processes")
    ap.add_argument("-v", "--verbose", required=False, default=True, type=bool, help="Verbose mode")
    args = vars(ap.parse_args())

    function_name_to_type = {'slic': SuperpixelSegmentationFunctionType.SLIC,
                             'felzenswalb': SuperpixelSegmentationFunctionType.FELZENSWALB,
                             'quickshift': SuperpixelSegmentationFunctionType.QUICKSHIFT,
                             'watershed': SuperpixelSegmentationFunctionType.WATERSHED}

    photos_path = args['photos']
    output_path = args['output']
    function_name = args['function']
    function_type = function_name_to_type.get(function_name, SuperpixelSegmentationFunctionType.NONE)
    n_jobs = args['jobs']
    verbose = args['verbose']

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print 'Reading photos from: {}'.format(photos_path)
    photos = image_utils.list_pictures(photos_path)
    print 'Found {} photos'.format(len(photos))

    print 'Ensuring the output directory: {} exists'.format(output_path)
    general_utils.create_path_if_not_existing(output_path)

    print 'Starting to create image masks with function: {}'.format(function_name)
    Parallel(n_jobs=n_jobs, backend='multiprocessing')\
        (delayed(generate_mask_for_unlabeled_image)(function_type, img_path, output_path, verbose) for img_path in photos)

    print 'Done'

if __name__ == '__main__':
    main()
