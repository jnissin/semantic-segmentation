"""
This program can be used to merge the masks in the IVRG dataset to produce similar
masks as in the Opensurfaces and MINC datasets.
"""

import numpy as np
import os
import sys
import multiprocessing

from joblib import Parallel, delayed
from keras.preprocessing.image import array_to_img, img_to_array, load_img


def get_bit(n, i):
    """
    Returns a boolen indicating whether the nth bit in the integer i was on or not.

    # Arguments
        n: index of the bit
        i: integer
    # Returns
        True if the nth bit was on false otherwise.
    """
    return n & (1 << i) > 0


def get_color_for_category_index(idx):
    """
    Returns a color for class index. The index is used in the red channel
    directly and the green and blue channels are derived similar to the
    PASCAL VOC coloring algorithm.

    # Arguments
        idx: Index of the class
    # Returns
        RGB color in channel range [0,255] per channel
    """

    cid = idx
    r = idx
    g = 0
    b = 0

    for j in range(0, 7):
        g = g | get_bit(cid, 0) << 7 - j
        b = b | get_bit(cid, 1) << 7 - j
        cid = cid >> 2

    color = np.array([r, g, b], dtype=np.uint8)
    return color


def np_replace_color(np_img, color, new_color):
    red, green, blue = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
    mask = (red == color[0]) & (green == color[1]) & (blue == color[2])
    np_img[:, :, :3][mask] = [new_color[0], new_color[1], new_color[2]]
    return np_img


def write_category_information_to_file(f, category_name, category_idx, category_red_color):
    f.write('{},{},{}\n'.format(category_idx, category_name, category_red_color))


def combine_mask(
        masks_directory_path,
        category_directories,
        processed_masks_directory_path,
        background_category_name,
        mask_file_name,
        category_to_color):

    # White color
    white_color = (255, 255, 255)

    # Load the background image
    bg_img = load_img(os.path.join(masks_directory_path, background_category_name, mask_file_name))
    np_bg_img = img_to_array(bg_img)

    # Go through all the category folders except the background directory
    # and try to find a matching mask file name
    for category in category_directories:
        category_files = os.listdir(os.path.join(masks_directory_path, category))
        category_idx_and_color = category_to_color[category]
        category_idx = category_idx_and_color[0]
        category_color = category_idx_and_color[1]

        #print 'Category {} with idx {} and color {}'.format(category, category_idx, category_color)

        if mask_file_name in category_files:
            category_img = load_img(os.path.join(masks_directory_path, category, mask_file_name))
            np_category_img = img_to_array(category_img)

            # Replace the white parts in the category img
            np_category_img = np_replace_color(np_category_img, white_color, category_color)

            # Replace the appropriate parts in the background img
            np_bg_img += np_category_img

    # Replace the background which is still white with the background color
    bg_color = category_to_color[background_category_name][1]
    np_bg_img = np_replace_color(np_bg_img, white_color, bg_color)

    # Write the file
    processed_img = array_to_img(np_bg_img, scale=False)
    processed_img.save(os.path.join(processed_masks_directory_path, mask_file_name), format='PNG')
    print '{} mask flattened'.format(mask_file_name)


"""
# Arguments
    1: directory of the masks
    2: directory of the processed masks
    3: name of the background category
"""
if __name__ == '__main__':

    if len(sys.argv) < 4:
        print 'Invalid number of arguments, use: python {} <masks dir> <processed masks dir> <bg category name>'
        sys.exit(1)

    masks_directory_path = sys.argv[1]
    processed_masks_directory_path = sys.argv[2]
    background_category_name = sys.argv[3]
    materials_file_name = 'ivrg-materials.csv'

    # Ignored files/directories
    ignored_files = ['.DS_Store']

    # Find the category directories
    category_directories = os.listdir(masks_directory_path)
    category_directories = [c for c in category_directories if c not in ignored_files and not c.startswith('.')]

    print 'Found {} category directories'.format(len(category_directories))

    # Create a dictionary and a CSV material file of
    # category name -> (idx, color)
    category_to_color = {}

    if background_category_name in category_directories:
        print 'Found background directory: {} in categories'.format(background_category_name)

        # Remove the background directory from the category directories and handle it as a
        # special case
        category_directories.remove(background_category_name)

    else:
        raise ValueError('Could not find background class: {} in category directories'.format(background_category_name))

    with open(os.path.join(processed_masks_directory_path, materials_file_name), 'w') as f:
        # Handle background separately
        f.write('substance_id,substance_name,red_color\n')
        category_to_color[background_category_name] = (0, get_color_for_category_index(0))
        write_category_information_to_file(f, background_category_name, 0, 0)

        for category_idx, category in enumerate(category_directories):
            cidx = category_idx+1
            category_color = get_color_for_category_index(cidx)
            category_to_color[category] = (cidx, category_color)
            write_category_information_to_file(f, category, cidx, category_color[0])
            print 'Category {} has idx {} and color {}'.format(category, cidx, category_color)

    print 'Material category information written to: {}'\
        .format(os.path.join(processed_masks_directory_path, materials_file_name))

    # Create matching combined mask files with the colors in the dictionary
    mask_files = os.listdir(os.path.join(masks_directory_path, background_category_name))

    print 'Flattening {} mask files'.format(len(mask_files))

    num_cores = multiprocessing.cpu_count()
    n_jobs = min(32, num_cores)

    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(combine_mask)(
            masks_directory_path,
            category_directories,
            processed_masks_directory_path,
            background_category_name,
            mask_file,
            category_to_color) for mask_file in mask_files)

    print 'Process complete'
