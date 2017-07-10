# coding=utf-8

import argparse
import os
import numpy as np

from keras.preprocessing.image import list_pictures, load_img, img_to_array, array_to_img


class Category(object):

    def __init__(self, name, red):
        # type: (str, int) -> ()

        self.name = name
        self.red = red

        # Calculate a unique color according to the red color, similar to
        # PASCAL VOC coloring scheme
        cid = red
        r = red
        g = 0
        b = 0

        for j in range(0, 7):
            g = g | get_bit(cid, 0) << 7 - j
            b = b | get_bit(cid, 1) << 7 - j
            cid = cid >> 2

        self.color = np.array([r, g, b], dtype=np.uint8)


class Segment(object):

    def __init__(self, file_path, photo_id, shape_id, category_idx):
        # type: (str, str, str, int) -> ()

        self.file_path = file_path
        self.photo_id = photo_id
        self.shape_id = shape_id
        self.category_idx = category_idx


class SegmentListEntry(object):

    def __init__(self, category_idx, photo_id, shape_id):
        # type: (int, str, str) -> ()

        self.category_idx = category_idx
        self.photo_id = photo_id
        self.shape_id = shape_id


def get_bit(n, i):
    # type: (int, int) -> bool

    """
    Returns a boolean indicating whether the nth bit in the integer i was on or not.

    # Arguments
        n: index of the bit
        i: integer
    # Returns
        True if the nth bit was on false otherwise.
    """
    return n & (1 << i) > 0


def combine_segments(categories, segments):
    # type: (list[Category], list[Segment]) -> np.array
    np_img = None

    for s in segments:
        segment_img = load_img(s.file_path)
        segment_img_array = img_to_array(segment_img)

        # If the image is none create a new image where everything is flattened
        if np_img is None:
            np_img = np.zeros(shape=segment_img_array.shape, dtype=np.uint8)

        mask = segment_img_array[:, :, 0] != 0
        np_img[:, :, :3][mask] = categories[s.category_idx].color

    return np_img


if __name__ == '__main__':

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to segmentation masks folder")
    ap.add_argument("-s", "--segments", required=True, help="Path to the segments list file")
    ap.add_argument("-c", "--categories", required=True, help="Path to the MINC categories file with red color mapping")
    ap.add_argument("-o", "--output", required=True, help="Path to the output folder")
    args = vars(ap.parse_args())

    segments_folder_path = args["path"]
    segments_list_file_path = args["segments"]
    categories_map_file_path = args["categories"]
    output_folder_path = args["output"]

    # Read the categories
    print "Reading the category -> red color mapping information from file: {}".format(categories_map_file_path)

    with open(categories_map_file_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    # Create category index to Category object mapping
    categories = []

    for idx, line in enumerate(content):
        name_and_red_color = line.split(',')
        categories.append(Category(name_and_red_color[0], int(name_and_red_color[1])))

    print 'Successfully read {} category mappings'.format(len(categories))

    # Read the segments list file and create a mapping shape_id -> SegmentListEntry
    print 'Reading segment list information from file: {}'.format(segments_list_file_path)

    with open(segments_list_file_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    shape_id_to_segment_list_entry = {}

    for idx, line in enumerate(content):
        # Each line is a tuple of: category_idx,photo_id,shape_id
        parts = line.split(',')
        category_idx = int(parts[0])
        photo_id = parts[1]
        shape_id = parts[2]

        # Assumes that shape ids are unique
        segment_list_entry = SegmentListEntry(category_idx, photo_id, shape_id)
        shape_id_to_segment_list_entry[shape_id] = segment_list_entry

    print 'Successfully created segment list mapping with {} entries'.format(len(shape_id_to_segment_list_entry))

    # Create a mapping of all the segments that belong to the same photo so photo_id -> Segment
    print 'Reading segment files'
    segment_files = list_pictures(segments_folder_path)
    print 'Found {} segment files'.format(len(segment_files))

    print 'Creating a photo id to segment mapping'
    photo_id_to_segments = {}

    for f in segment_files:
        # Parse the photo and shape id from the file name which is: PHOTOID_SHAPEID.png
        file_name = os.path.basename(f)
        file_name_no_ext = file_name[:-4]

        photoid_and_shapeid = file_name_no_ext.split('_')
        photo_id = photoid_and_shapeid[0]
        shape_id = photoid_and_shapeid[1]

        # Get the category idx
        category_idx = shape_id_to_segment_list_entry[shape_id].category_idx

        segment = Segment(f, photo_id, shape_id, category_idx)

        # If the photo id is not in the dictionary create a new list with the photo id
        # to the dictionary
        if not photo_id_to_segments.has_key(photo_id):
            photo_id_to_segments[photo_id] = []

        photo_id_to_segments[photo_id].append(segment)

    # Use the categories and the photo id -> segment mapping to create new flattened images
    print 'Starting to flatten segments to {} segmentation mask images'.format(len(photo_id_to_segments))

    for key, value in photo_id_to_segments.iteritems():
        print 'Processing photo id: {} with {} segments'.format(key, len(value))
        img_array = combine_segments(categories, value)
        img = array_to_img(img_array, scale=False)
        img.save(os.path.join(output_folder_path, key + '.png'))

    print 'Done'
