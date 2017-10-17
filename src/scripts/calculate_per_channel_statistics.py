import argparse
import os

from ..utils import dataset_utils
from ..data_set import ImageFile


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
    ap.add_argument("-s", "--segments", required=True, type=str, help="The segmentation data set JSON file")
    ap.add_argument("-p", "--patches", required=True, type=str, help="The patches training set text file")
    ap.add_argument("-u", "--unlabeled", required=True, type=str, help="Path to the unlabeled photo files")
    args = vars(ap.parse_args())

    segments_data_set_json_path = args['segments']
    patches_training_set_path = args['patches']
    unlabeled_path = args['unlabeled']

    """
    Parse the labeled training set from the segments and patches training set data
    """

    print 'Reading segments data set information from: {}'.format(segments_data_set_json_path)
    segments_data_set_information = dataset_utils.load_segmentation_data_set_information(segments_data_set_json_path)
    print 'Read segments data set information successfully'

    num_training_set_segments = len(segments_data_set_information.training_set.labeled_photos)
    print 'Segments training set size: {}'.format(num_training_set_segments)

    print 'Reading patches training set information from: {}'.format(patches_training_set_path)
    patch_photo_ids = get_minc_photo_ids_in_set(patches_training_set_path)
    print 'Found {} patch photo ids'.format(len(patch_photo_ids))
    patch_photo_ids = list(set(patch_photo_ids))
    patch_photo_ids_short = [str(int(photo_id)) for photo_id in patch_photo_ids]
    print 'Found {} unique patch photo ids'.format(len(patch_photo_ids))

    segment_photo_ids = [photo_name.split('.')[0] for photo_name in segments_data_set_information.training_set.labeled_photos]
    segment_photo_ids = set(segment_photo_ids)
    segment_photos_ids_not_in_patch_photos = segment_photo_ids - (set(patch_photo_ids) | set(patch_photo_ids_short))

    labeled_photos = []

    for photo_id in patch_photo_ids:
        labeled_photos.append('/Volumes/Omenakori/data/final/labeled/patches/photos/{}.jpg'.format(photo_id))

    for photo_id in segment_photos_ids_not_in_patch_photos:
        labeled_photos.append('/Volumes/Omenakori/data/final/labeled/segments/photos/{}.jpg'.format(photo_id))

    print 'Found {} unique training set labeled photos (patches + segments)'.format(len(labeled_photos))


    """
    Read the unlabeled photos
    """

    print 'Reading unlabeled photos from {}'.format(unlabeled_path)
    unlabeled_photos = get_files(unlabeled_path)
    print 'Found {} unlabeled photos'.format(len(unlabeled_photos))
    print 'Found unique {} photos in training set'.format(len(labeled_photos) + len(unlabeled_photos))

    """
    Calculate per channel mean for labeled and unlabeled
    """

    print 'Creating labeled ImageFile instances'
    labeled_image_files = [ImageFile(image_path=image_path) for image_path in labeled_photos]
    unlabeled_image_files = [ImageFile(image_path=image_path) for image_path in unlabeled_photos]
    all_image_files = labeled_image_files + unlabeled_image_files

    print 'Starting to calculate labeled per-channel mean for {} labeled images'.format(len(labeled_image_files))
    labeled_per_channel_mean = dataset_utils.calculate_per_channel_mean(labeled_image_files, num_channels=3, verbose=True)
    print 'Labeled per-channel mean: {}'.format(list(labeled_per_channel_mean))

    print 'Starting to calculate labeled per-channel stddev {} labeled images'.format(len(labeled_image_files))
    labeled_per_channel_stddev = dataset_utils.calculate_per_channel_stddev(labeled_image_files, per_channel_mean=labeled_per_channel_mean, num_channels=3, verbose=True)
    print 'Labeled per-channel stddev: {}'.format(list(labeled_per_channel_stddev))

    print 'Starting to calculate labeled+unlabeled per-channel mean {} images'.format(len(all_image_files))
    unlabeled_per_channel_mean = dataset_utils.calculate_per_channel_mean(all_image_files, num_channels=3, verbose=True)
    print 'Unlabeled per-channel mean: {}'.format(list(unlabeled_per_channel_mean))

    print 'Starting to calculate labeled+unlabeled per-channel stddev {} images'.format(len(all_image_files))
    unlabeled_per_channel_stddev = dataset_utils.calculate_per_channel_stddev(all_image_files, per_channel_mean=unlabeled_per_channel_mean, num_channels=3, verbose=True)
    print 'Unlabeled per-channel stddev: {}'.format(list(unlabeled_per_channel_stddev))

    print 'Labeled per-channel mean: {}'.format(list(labeled_per_channel_mean))
    print 'Labeled per-channel stddev: {}'.format(list(labeled_per_channel_stddev))
    print 'Unlabeled per-channel mean: {}'.format(list(unlabeled_per_channel_mean))
    print 'Unlabeled per-channel stddev: {}'.format(list(unlabeled_per_channel_stddev))

    print 'Done'

if __name__ == '__main__':
    main()
