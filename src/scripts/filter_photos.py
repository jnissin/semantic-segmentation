# coding=utf-8

'''
This script filters all the photos and masks in which the percentage of background
pixels is above a certain threshold
'''

import os
import multiprocessing
import time
import argparse

import numpy as np

from PIL import Image
from joblib import Parallel, delayed
from keras.preprocessing.image import load_img, img_to_array

NUM_FILTERED = 0


def get_files(path, ignore_hidden_files=True, include_sub_dirs=False):
    ret_files = []

    if not include_sub_dirs:
        # Files under the directory directly
        ret_files = os.listdir(path)

        # Filter the hidden files out
        if ignore_hidden_files:
            ret_files = [f for f in ret_files if not f.startswith('.')]

        # Complete the file paths and check that we are only returning files
        ret_files = [os.path.join(path, f) for f in ret_files]
        ret_files = [f for f in ret_files if os.path.isfile(os.path.join(path, f))]
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if ignore_hidden_files and name.startswith('.'):
                    continue

                file_path = os.path.join(root, name)
                if os.path.isfile(file_path) and not name.startswith('.'):
                    ret_files.append(file_path)

    return ret_files


def filter_pair(threshold, photo_file_path, mask_file_path, path_to_filtered_photos, path_to_filtered_masks, dryrun, inplace):

    photo_file_name = os.path.basename(photo_file_path)
    photo_file_name_no_ext = photo_file_name.split('.')[0]
    mask_file_name = os.path.basename(mask_file_path)
    mask_file_name_no_ext = mask_file_name.split('.')[0]

    if photo_file_name_no_ext != mask_file_name_no_ext:
        raise ValueError('Unmatching photo and mask filenames: {} vs {}'.format(photo_file_name_no_ext, mask_file_name_no_ext))

    start_time = time.time()

    # Load the mask image and count the percentage of background pixels
    mask_img = load_img(mask_file_path)
    mask_img_array = img_to_array(mask_img)

    background_pixel_mask = mask_img_array[:, :, 0] == 0.0
    num_background_pixels = np.sum(background_pixel_mask, axis=(0, 1))
    num_pixels = mask_img_array.shape[0]*mask_img_array.shape[1]
    proportion_background_pixels = float(num_background_pixels)/float(num_pixels)

    # If there are more background pixels than the threshold
    # save the photo and mask
    if proportion_background_pixels > threshold:
        global NUM_FILTERED
        NUM_FILTERED += 1

        photo = Image.open(photo_file_path)

        if not dryrun:
            if not inplace:
                photo.save(os.path.join(path_to_filtered_photos, photo_file_name))
                mask_img.save(os.path.join(path_to_filtered_masks, mask_file_name))
            else:
                os.remove(photo_file_path)
                os.remove(mask_file_path)

        print 'Filtering of photo {} with {}% background pixels completed in {} sec - filtered: {}'.format(
            photo_file_name,
            proportion_background_pixels*100.0,
            time.time()-start_time,
            proportion_background_pixels > threshold)


def main():

    ap = argparse.ArgumentParser(description="Creates a new filtered set of all the images where the proportion of background pixels is less than specified")
    ap.add_argument("-p", "--photos", required=True, help="Path to photos folder")
    ap.add_argument("-m", "--masks", required=True, help="Path to masks folder")
    ap.add_argument("-o", "--output", required=False, help="Output folder, will create two subfolders under the folder for photos and masks")
    ap.add_argument("-t", "--threshold", required=True, type=float, help="Threshold for the percentage of background pixels images with values > threshold will be filtered")
    ap.add_argument("-d", "--dryrun", required=False, default=False, type=bool, help="Should we skip saving the dataset to disk")
    ap.add_argument("-i", "--inplace", required=False, default=False, type=bool, help="Should we just delete the files instead of creating a new dataset")
    args = vars(ap.parse_args())

    path_to_photos = args['photos']
    path_to_masks = args['masks']
    path_to_output = args['output']
    threshold = args['threshold']
    dryrun = args['dryrun']
    inplace = args['inplace']

    if path_to_output is None and not inplace:
        raise ValueError('Either output has to be specified or inplace has to be true')

    photo_files = get_files(path_to_photos)
    mask_files = get_files(path_to_masks)

    if len(photo_files) != len(mask_files):
        raise ValueError('Unmatching photo and mask file dataset sizes: {} vs {}'
                         .format(len(photo_files), len(mask_files)))

    path_to_filtered_photos = None
    path_to_filtered_masks = None

    if path_to_output is not None:
        path_to_filtered_photos = os.path.join(path_to_output, 'photos')
        path_to_filtered_masks = os.path.join(path_to_output, 'masks')

        if os.path.exists(path_to_filtered_photos) or os.path.exists(path_to_filtered_photos):
            raise ValueError('Path for masks/photos in output already exists: {}, {}'
                .format(path_to_filtered_photos, path_to_filtered_masks))

        os.mkdir(path_to_filtered_photos)
        os.mkdir(path_to_filtered_masks)

    # Start job

    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    print 'Starting with {} processes with threshold {}, inplace: {}, dryrun: {}, for {} photos and masks'\
        .format(n_jobs, threshold, inplace, dryrun, len(photo_files))

    res = raw_input('Continue (Y/N)? ')
    if str(res).lower() != 'y':
        exit(0)

    Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(filter_pair)(
            threshold,
            photo_files[i],
            mask_files[i],
            path_to_filtered_photos,
            path_to_filtered_masks,
            dryrun,
            inplace) for i in range(0, len(photo_files)))

    global NUM_FILTERED
    new_dataset_percentage = (float(NUM_FILTERED)/float(len(photo_files))) * 100.0
    print 'Run finished with background proportion threshold of {}.'.format(threshold)
    print 'Filtered dataset size: {} which is {}% of the original dataset'.format(NUM_FILTERED, new_dataset_percentage)


if __name__ == '__main__':
    main()
