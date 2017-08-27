# coding=utf-8

import argparse
import os
import time
import json
import numpy as np

from keras.preprocessing.image import list_pictures, load_img, img_to_array
from joblib import Parallel, delayed

from ..utils import dataset_utils

class MaterialStatistics(object):

    def __init__(self,
                 image_statistics):
        # type: (list[ImageStatistics]) -> ()

        self.image_statistics = image_statistics

        # Calculate total number of pixels
        self.total_pixels = sum([s.num_pixels for s in self.image_statistics])

        # Calculate total number of pixels of different materials
        self.material_pixels = [s.material_pixels for s in self.image_statistics]
        self.material_pixels = zip(*self.material_pixels)
        self.material_pixels = [sum(x) for x in self.material_pixels]

        # Calculate total number of occurences of different materials and record
        # the files that have instances of each material
        self.material_occurences = [0] * len(self.material_pixels)
        self.material_occurence_files = [[] for i in range(len(self.material_pixels))]

        for s in self.image_statistics:
            for i in range(0, len(s.material_pixels)):
                if s.material_pixels[i] != 0:
                    self.material_occurences[i] += 1
                    self.material_occurence_files[i].append(s.image_name)

class ImageStatistics(object):

    def __init__(self, image_name, num_pixels, material_pixels):
        # type: (str, int, list[int]) -> ()

        self.image_name = image_name
        self.num_pixels = num_pixels
        self.material_pixels = material_pixels


def calculate_image_statistics(mask_file_path, materials):
    # type: (str, list[MaterialClassInformation]) -> ImageStatistics

    mask_img = load_img(mask_file_path)
    mask_img_array = img_to_array(mask_img)
    expanded_mask = dataset_utils.one_hot_encode_mask(mask_img_array, materials)

    image_name = os.path.basename(mask_file_path)
    num_pixels = mask_img_array.shape[0] * mask_img_array.shape[1]
    material_pixels = []

    for i in range(0, expanded_mask.shape[2]):
        material_pixels.append(float(np.sum(expanded_mask[:, :, i])))

    img_stats = ImageStatistics(image_name, num_pixels, material_pixels)
    return img_stats


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to segmentation masks folder")
    ap.add_argument("-m", "--materials", required=True, help="Path to materials CSV file")
    ap.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    args = vars(ap.parse_args())

    masks_path = args["path"]
    materials_path = args["materials"]
    output_path = args["output"]

    print 'Loading material information from file: {}'.format(materials_path)
    materials = dataset_utils.load_material_class_information(materials_path)
    print 'Loaded {} materials'.format(len(materials))

    print 'Reading masks from directory: {}'.format(masks_path)
    masks = list_pictures(masks_path)
    print 'Found {} mask images'.format(len(masks))

    n_jobs = dataset_utils.get_number_of_parallel_jobs()

    print 'Starting per-image material statistics calculation with {} jobs'.format(n_jobs)

    start_time = time.time()
    image_stats = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(calculate_image_statistics)(f, materials) for f in masks)

    print 'Per-image material statistics calculation of {} files finished in {} seconds'\
        .format(len(image_stats), time.time()-start_time)

    material_stats = MaterialStatistics(image_stats)

    print 'Writing material statistics to file: {}'.format(output_path)
    json_str = json.dumps(material_stats, default=lambda o: o.__dict__)

    with open(output_path, 'w') as f:
        f.write(json_str)

    print 'Material statistics stored successfully'


if __name__ == '__main__':
    main()