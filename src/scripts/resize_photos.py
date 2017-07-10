'''
This script resizes all the photos to match the mask sizes or a given smaller dimension.
'''

import os
import multiprocessing
import time
import argparse

from PIL import Image, ImageFile
from joblib import Parallel, delayed


def get_files(path, ignore_hidden_files=True):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if ignore_hidden_files:
        files = [f for f in files if not f.startswith('.')]
    return files


def resize_to_match(path_to_photos, photo_filename, path_to_masks, mask_filename, path_to_resized_folder):
    if photo_filename.split('.')[0] != mask_filename.split('.')[0]:
        raise ValueError('Unmatching photo and mask filenames: {} vs {}'.format(photo_filename.split('.')[0],
                                                                                mask_filename.split('.')[0]))

    photo = Image.open(os.path.join(path_to_photos, photo_filename))
    mask = Image.open(os.path.join(path_to_masks, mask_filename))
    resize(photo, photo_filename, mask.size, path_to_resized_folder)


def resize_to_smaller_dimension(path_to_photos, photo_filename, smaller_dimension, path_to_resized_folder):
    photo = Image.open(os.path.join(path_to_photos, photo_filename))

    # Calculate the target size so the smaller dimension matches the specified
    scale_factor = float(smaller_dimension)/float(min(photo.width, photo.height))
    target_width = int(round(scale_factor * photo.width))
    target_height = int(round(scale_factor * photo.height))
    resize(photo, photo_filename, (target_width, target_height), path_to_resized_folder)


def resize(photo, photo_filename, target_size, path_to_resized_folder):
    start_time = time.time()
    original_size = photo.size

    if photo.size == target_size:
        photo.save(os.path.join(path_to_resized_folder, photo_filename))
    else:
        photo = photo.resize(target_size, Image.ANTIALIAS)
        photo.save(os.path.join(path_to_resized_folder, photo_filename))

    if photo.size != target_size:
        raise ValueError('Unmatching photo and target sizes even after resizing: {} vs {}', photo.size, target_size)

    dt = time.time() - start_time
    print 'Resizing of image {} from {} to {} completed in {} sec'.format(photo_filename, original_size, photo.size, dt)


def main():

    ap = argparse.ArgumentParser(description="Resizes a folder of photos match masks sizes or to given smaller dimension")
    ap.add_argument("-p", "--photos", required=True, help="Path to photos folder")
    ap.add_argument("-m", "--masks", required=False, help="Path to masks folder")
    ap.add_argument("-d", "--sdim", required=False, type=int, help="Size of the smaller dimension")
    ap.add_argument("-o", "--output", required=True, help="Path to the output folder")
    args = vars(ap.parse_args())

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    path_to_photos = args['photos']
    path_to_masks = args['masks']
    smaller_dim = args['sdim']
    path_to_output = args['output']
    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    if path_to_masks is not None:
        photo_files = get_files(path_to_photos)
        mask_files = get_files(path_to_masks)

        if len(photo_files) != len(mask_files):
            raise ValueError('Unmatching photo and mask file dataset sizes: {} vs {}'.format(len(photo_files), len(mask_files)))

        Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(resize_to_match)(
                path_to_photos,
                photo_files[i],
                path_to_masks,
                mask_files[i],
                path_to_output) for i in range(0, len(photo_files)))
    elif smaller_dim is not None:
        photo_files = get_files(path_to_photos)

        Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(resize_to_smaller_dimension)(
                path_to_photos,
                photo_files[i],
                smaller_dim,
                path_to_output) for i in range(0, len(photo_files)))
    else:
        raise RuntimeError('You must provide either sdim or masks to resize')


if __name__ == '__main__':
    main()
