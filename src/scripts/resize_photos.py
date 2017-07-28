# coding=utf-8

"""
This script resizes all the photos to match the mask sizes or a given smaller dimension.
"""

import os
import time
import argparse

from PIL import Image, ImageFile
from joblib import Parallel, delayed

from ..utils import dataset_utils


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


def resize_to_match(photo_file_path,
                    mask_file_path,
                    path_to_resized_folder,
                    resampling=Image.ANTIALIAS,
                    ignore_existing=False,
                    inplace=False):

    photo_filename = os.path.basename(photo_file_path)
    mask_filename = os.path.basename(mask_file_path)

    if photo_filename.split('.')[0] != mask_filename.split('.')[0]:
        raise ValueError('Unmatching photo and mask filenames: {} vs {}'.format(photo_filename.split('.')[0],
                                                                                mask_filename.split('.')[0]))

    photo = Image.open(photo_file_path)
    mask = Image.open(mask_file_path)
    resize(photo, photo_file_path, mask.size, path_to_resized_folder, resampling, ignore_existing, inplace)


def resize_to_smaller_dimension(photo_file_path,
                                smaller_dimension,
                                path_to_resized_folder,
                                resampling=Image.ANTIALIAS,
                                ignore_existing=False,
                                inplace=False):

    photo = Image.open(photo_file_path)

    # Calculate the target size so the smaller dimension matches the specified
    scale_factor = float(smaller_dimension)/float(min(photo.width, photo.height))
    target_width = int(round(scale_factor * photo.width))
    target_height = int(round(scale_factor * photo.height))
    resize(photo, photo_file_path, (target_width, target_height), path_to_resized_folder, resampling, ignore_existing, inplace)


def resize(photo, photo_file_path, target_size, path_to_resized_folder, resampling=Image.ANTIALIAS, ignore_existing=False, inplace=False):
    start_time = time.time()
    original_size = photo.size
    photo_file_name = os.path.basename(photo_file_path)

    if not inplace:
        resized_file_path = os.path.join(path_to_resized_folder, photo_file_name)

        if ignore_existing and os.path.exists(resized_file_path):
            print 'Skipping existing: {}'.format(photo_file_name)
            return

        if photo.size == target_size:
            photo.save(resized_file_path)
        else:
            photo = photo.resize(target_size, resampling)

        photo.save(resized_file_path)
    else:
        if photo.size == target_size:
            print 'Image {} already at target size: {}'.format(photo_file_name, target_size)
            return

        photo = photo.resize(target_size, resampling)
        photo.save(photo_file_path)

    if photo.size != target_size:
        raise ValueError('Unmatching photo and target sizes even after resizing: {} vs {}', photo.size, target_size)

    dt = time.time() - start_time
    print 'Resizing of image {} from {} to {} completed in {} sec'.format(photo_file_name, original_size, photo.size, dt)


def main():

    ap = argparse.ArgumentParser(description="Resizes a folder of photos match masks sizes or to given smaller dimension")
    ap.add_argument("-p", "--photos", required=True, help="Path to photos folder")
    ap.add_argument("-m", "--masks", required=False, help="Path to masks folder")
    ap.add_argument("-d", "--sdim", required=False, type=int, help="Size of the smaller dimension")
    ap.add_argument("-o", "--output", required=False, help="Path to the output folder")
    ap.add_argument("-x", "--inplace", required=False, default=False, type=bool, help="Should the resize happen in place")
    ap.add_argument("-i", "--incsub", required=False, default=False, type=bool, help="Include sub directories (default false)")
    ap.add_argument("-r", "--resampling", required=False, default='nearest', type=str, help="Resampling method (nearest or antialias)")
    ap.add_argument("-s", "--skip", required=False, default=False, type=bool, help="Ignore existing (default false)")
    args = vars(ap.parse_args())

    # Without this some truncated images can throw errors
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    path_to_photos = args['photos']
    path_to_masks = args['masks']
    smaller_dim = args['sdim']
    path_to_output = args['output']
    include_sub_dirs = args['incsub']
    ignore_existing = args['skip']
    inplace = args['inplace']
    resampling = args['resampling']

    resampling = Image.NEAREST if resampling.lower() == 'nearest' else Image.ANTIALIAS
    print 'Using anti-alias for resampling: {}'.format(resampling == Image.ANTIALIAS)

    photo_files = get_files(path_to_photos, include_sub_dirs=include_sub_dirs)
    num_all_photo_files = len(photo_files)

    # Quickly filter out the existing files using sets
    if ignore_existing:
        existing_photo_files = get_files(path_to_output, include_sub_dirs=include_sub_dirs)
        tmp = [os.path.join(path_to_photos, os.path.basename(f)) for f in existing_photo_files]
        photo_files = list(set(photo_files).difference(set(tmp)))
        print 'Skipped {} existing photo files, {} photo files remain'.format(len(existing_photo_files), len(photo_files))

    print 'Starting resizing process for: {} photos'.format(len(photo_files))

    if not inplace and path_to_output is None:
        raise ValueError('Either inplace has to be true or output has to be specified')

    if path_to_masks is not None:
        mask_files = get_files(path_to_masks, include_sub_dirs=include_sub_dirs)

        if num_all_photo_files != len(mask_files):
            raise ValueError('Unmatching photo and mask file dataset sizes: {} vs {}'.format(len(photo_files), len(mask_files)))

        Parallel(n_jobs=dataset_utils.get_number_of_parallel_jobs(), backend='multiprocessing')(
            delayed(resize_to_match)(
                photo_files[i],
                mask_files[i],
                path_to_output,
                resampling=resampling,
                ignore_existing=ignore_existing,
                inplace=inplace) for i in range(0, len(photo_files)))

    elif smaller_dim is not None:
        Parallel(n_jobs=dataset_utils.get_number_of_parallel_jobs(), backend='multiprocessing')(
            delayed(resize_to_smaller_dimension)(
                photo_files[i],
                smaller_dim,
                path_to_output,
                resampling=resampling,
                ignore_existing=ignore_existing,
                inplace=inplace) for i in range(0, len(photo_files)))

    else:
        raise RuntimeError('You must provide either sdim or masks to resize')


if __name__ == '__main__':
    main()
