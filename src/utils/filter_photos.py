'''
This script finds all the images in which the percentage of background
pixels is above a certain threshold
'''

import os
import multiprocessing
import time

import numpy as np

from PIL import Image
from joblib import Parallel, delayed
from keras.preprocessing.image import load_img, img_to_array

PATH_TO_PHOTOS = '/Volumes/Omenakori/opensurfaces/photos-resized/'
PATH_TO_MASKS = '/Volumes/Omenakori/opensurfaces/photos-labels/'

PATH_TO_FILTERED_PHOTOS = '/Volumes/Omenakori/opensurfaces/photos-filtered/'
PATH_TO_FILTERED_MASKS = '/Volumes/Omenakori/opensurfaces/masks-filtered/'

DRY_RUN = False
THRESHOLD = 0.80

NUM_FILTERED = 0


def get_files(path, ignore_hidden_files=True):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if ignore_hidden_files:
        files = [f for f in files if not f.startswith('.')]
    return files


def filter_pair(
    threshold,
    path_to_photos,
    photo_filename,
    path_to_masks,
    mask_filename,
    path_to_filtered_photos,
    path_to_filtered_masks):

    if (photo_filename.split('.')[0] != mask_filename.split('.')[0]):
        raise ValueError('Unmatching photo and mask filenames: {} vs {}'
                         .format(photo_filename.split('.')[0], mask_filename.split('.')[0]))

    start_time = time.time()

    # Load the mask image and count the percentage of background pixels
    mask_img = load_img(os.path.join(path_to_masks, mask_filename))
    mask_img_array = img_to_array(mask_img)

    background_pixel_mask = mask_img_array[:, :, 0] == 0.0

    num_background_pixels = np.sum(background_pixel_mask, axis=(0, 1))
    proportion_background_pixels = float(num_background_pixels)/float(mask_img_array.shape[0]*mask_img_array.shape[1])

    # If there are less background pixels than the threshold
    # save the photo and mask
    if (proportion_background_pixels < threshold):
        global NUM_FILTERED
        NUM_FILTERED += 1

        photo = Image.open(os.path.join(path_to_photos, photo_filename))
        if not DRY_RUN:
            photo.save(os.path.join(path_to_filtered_photos, photo_filename))
            mask_img.save(os.path.join(path_to_filtered_masks, mask_filename))

    print 'Filtering of image {} with {}% background pixels completed in {} sec - filtered: {}'.format(
        photo_filename,
        proportion_background_pixels*100.0,
        time.time()-start_time,
        proportion_background_pixels > threshold)


if __name__ == '__main__':

    photo_files = get_files(PATH_TO_PHOTOS)
    mask_files = get_files(PATH_TO_MASKS)

    if len(photo_files) != len(mask_files):
        raise ValueError('Unmatching photo and mask file dataset sizes: {} vs {}'.format(len(photo_files), len(mask_files)))

    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(filter_pair)(
            THRESHOLD,
            PATH_TO_PHOTOS,
            photo_files[i],
            PATH_TO_MASKS,
            mask_files[i],
            PATH_TO_FILTERED_PHOTOS,
            PATH_TO_FILTERED_MASKS) for i in range(0, len(photo_files)))

    new_dataset_percentage = (float(NUM_FILTERED)/float(len(photo_files))) * 100.0

    print 'Run finished with background proportion threshold of {}.'.format(THRESHOLD)
    print 'Filtered dataset size: {} which is {}% of the original dataset'.format(NUM_FILTERED, new_dataset_percentage)