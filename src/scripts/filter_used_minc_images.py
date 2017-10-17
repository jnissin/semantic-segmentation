# coding=utf-8

"""
This script filters all the images used by the MINC patch data from a given folder.
"""

import argparse
import os


def get_files(path, ignore_hidden_files=True, include_sub_dirs=False):

    if not include_sub_dirs:
        # Files under the directory directly
        ret_files = os.listdir(path)

        # Filter the hidden files out
        if ignore_hidden_files:
            ret_files = [f for f in ret_files if not f.startswith('.')]

        # Complete the file paths and check that we are only returning files
        ret_files = [os.path.join(path, f) for f in ret_files]
        ret_files = [f for f in ret_files if os.path.isfile(f)]
        return ret_files
    else:
        ret_files = []

        for root, dirs, files in os.walk(path):
            for name in files:
                if ignore_hidden_files and name.startswith('.'):
                    continue

                file_path = os.path.join(root, name)
                if os.path.isfile(file_path) and not name.startswith('.'):
                    ret_files.append(file_path)

        return ret_files


def get_minc_photo_ids_in_set(path_to_set):
    with open(path_to_set) as f:
        lines = f.readlines()

    photo_ids = []

    for line in lines:
        line = line.strip()

        # Filter any empty lines
        if len(line) == 0:
            continue

        # Each line is a 4-tuple list of (label, photo id, x, y)
        parts = line.split(',')

        if len(parts) != 4:
            raise ValueError('Invalid data, expected 4 parts got: {}'.format(len(parts)))

        photo_id = parts[1].strip()
        photo_ids.append(photo_id)

    return photo_ids


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=False, type=str, help="Path to the MINC segments photos")
    ap.add_argument("--trainingset", required=False, type=str, help="Path to the MINC training set file")
    ap.add_argument("--validationset", required=False, type=str, help="Path to the MINC validation set file")
    ap.add_argument("--testset", required=False, type=str, help="Path to the MINC test set file")
    ap.add_argument("--photos", required=True, type=str, help="Path to the MINC photos path")
    ap.add_argument("--move", required=False, type=str, help="Path to move the used images")
    args = vars(ap.parse_args())

    segments_photos_path = args['segments']
    training_set_path = args['trainingset']
    validation_set_path = args['validationset']
    test_set_path = args['testset']
    photos_path = args['photos']
    move_path = args['move']

    if segments_photos_path is None and (training_set_path is None or validation_set_path is None or test_set_path is None):
        raise ValueError('Must provide either patches training set files or segments photos path')

    training_set_photo_ids = []
    validation_set_photo_ids = []
    test_set_photo_ids = []
    segments_photo_ids = []

    if training_set_path is not None:
        print 'Reading training set photos from file: {}'.format(training_set_path)
        training_set_photo_ids = get_minc_photo_ids_in_set(training_set_path)
        print 'Number of training set samples: {}'.format(len(training_set_photo_ids))

    if validation_set_path is not None:
        print 'Reading validation set photos from file: {}'.format(validation_set_path)
        validation_set_photo_ids = get_minc_photo_ids_in_set(validation_set_path)
        print 'Number of validation set samples: {}'.format(len(validation_set_photo_ids))

    if test_set_path is not None:
        print 'Reading test set photos from file: {}'.format(test_set_path)
        test_set_photo_ids = get_minc_photo_ids_in_set(test_set_path)
        print 'Number of test set samples: {}'.format(len(test_set_photo_ids))

    if segments_photos_path is not None:
        print 'Reading segments photo files from: {}'.format(segments_photos_path)
        segments_photo_ids = get_files(segments_photos_path, ignore_hidden_files=True)
        print 'Found {} photo files'.format(len(segments_photo_ids))
        segments_photo_ids = [os.path.basename(photo_path).split('.')[0] for photo_path in segments_photo_ids]

    all_photo_ids = training_set_photo_ids + validation_set_photo_ids + test_set_photo_ids + segments_photo_ids
    num_samples = len(all_photo_ids)
    unique_photo_ids = list(set(all_photo_ids))
    num_unique_photo_ids = len(unique_photo_ids)

    print 'Found {} unique photo ids from a total of {} samples'.format(num_unique_photo_ids, num_samples)

    # Read the files
    print 'Reading photo files from: {}'.format(photos_path)
    photos = get_files(photos_path, ignore_hidden_files=True)
    print 'Found {} photos from the directory'.format(len(photos))

    photo_file_names = [os.path.basename(photo) for photo in photos]
    photo_file_name_look_up_table = {}

    for idx, photo_file_name in enumerate(photo_file_names):
        photo_file_name_no_ext = photo_file_name.split('.')[0]
        photo_file_name_short_no_ext = str(int(photo_file_name_no_ext))
        photo_file_name_look_up_table[photo_file_name_no_ext] = idx
        photo_file_name_look_up_table[photo_file_name_short_no_ext] = idx

    photos_to_be_removed = []
    no_photo_found = []

    for photo_id in unique_photo_ids:
        original_id = photo_id
        if original_id in photo_file_name_look_up_table:
            photos_to_be_removed.append(photos[photo_file_name_look_up_table[original_id]])
            continue

        short_id = str(int(photo_id))
        if short_id in photo_file_name_look_up_table:
            photos_to_be_removed.append(photos[photo_file_name_look_up_table[short_id]])
            continue

        no_photo_found.append(original_id)

    print 'Number of not found photos: {}'.format(len(no_photo_found))

    if len(no_photo_found) > 0:
        response = raw_input('Print the not found photos? (y/n): ')

        if str(response).lower() == 'y':
            print no_photo_found

    # Report how many found photos was there
    num_photos_to_be_removed = len(photos_to_be_removed)

    if move_path:
        response = raw_input('Found {} photos to be moved to: {}, would you like to continue (y/n): '.format(num_photos_to_be_removed, move_path))

        if not os.path.exists(os.path.dirname(move_path)):
            print 'Creating directory: {}'.format(move_path)
            os.mkdir(move_path)

        if str(response).lower() == 'y':
            for f in photos_to_be_removed:
                new_path = os.path.join(move_path, os.path.basename(f))
                print 'Moving from: {} to: {}'.format(f, new_path)
                os.rename(f, new_path)
    else:
        response = raw_input('Found {} photos to be removed, would you like to continue (y/n): '.format(num_photos_to_be_removed))

        if str(response).lower() == 'y':
            for f in photos_to_be_removed:
                print 'Removing: {}'.format(f)
                os.remove(f)

    print 'Done'

if __name__ == '__main__':
    main()
