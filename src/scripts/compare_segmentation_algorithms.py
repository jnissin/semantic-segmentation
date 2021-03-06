# coding=utf-8

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import skimage.exposure as exposure

from PIL import Image

from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import img_to_array

from ..utils import image_utils


def main():

    ap = argparse.ArgumentParser(description="Runs different segmentation methods for a single image and displays the results")
    ap.add_argument("-i", "--image", required=True, help="Path to image file")
    args = vars(ap.parse_args())

    image_path = args['image']

    img = Image.open(image_path)
    np_img = img_to_array(img)
    #np_img = exposure.equalize_hist(np_img.astype(np.uint8))
    np_img = image_utils.np_normalize_image_channels(np_img, clamp_to_range=True)
    print 'Image size: {}'.format(img.size)

    fz_time = time.time()
    segments_fz = image_utils.np_get_felzenszwalb_segmentation(np_img, scale=700, sigma=0.6, min_size=250)
    fz_time = time.time() - fz_time

    slic_time = time.time()
    segments_slic = image_utils.np_get_slic_segmentation(np_img, n_segments=300, compactness=10, sigma=1, max_iter=20)
    slic_time = time.time() - slic_time

    quick_time = time.time()
    segments_quick = image_utils.quickshift(np_img, kernel_size=5, max_dist=5, ratio=0.5)
    quick_time = time.time() - quick_time

    watershed_time = time.time()
    segments_watershed = image_utils.np_get_watershed_segmentation(np_img, markers=250, compactness=0.001)
    watershed_time = time.time() - watershed_time

    print("Felzenszwalb number of segments: {}, time: {}s".format(np.max(segments_fz), fz_time))
    print("SLIC number of segments: {}, time: {}s".format(np.max(segments_slic), slic_time))
    print("Quickshift number of segments: {}, time: {}s".format(np.max(segments_quick), quick_time))
    print("Watershed number of segments: {}, time: {}s".format(np.max(segments_watershed), watershed_time))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})

    boundary_mode = 'thick'

    ax[0, 0].imshow(mark_boundaries(img, segments_fz, mode=boundary_mode))
    ax[0, 0].set_title("Felzenszwalb, {}, {}s".format(len(np.unique(segments_fz)), fz_time))
    ax[0, 1].imshow(mark_boundaries(img, segments_slic, mode=boundary_mode))
    ax[0, 1].set_title("SLIC, {}, {}s".format(len(np.unique(segments_slic)), slic_time))
    ax[1, 0].imshow(mark_boundaries(img, segments_quick, mode=boundary_mode))
    ax[1, 0].set_title("Quickshift, {}, {}s".format(len(np.unique(segments_quick)), quick_time))
    ax[1, 1].imshow(mark_boundaries(img, segments_watershed, mode=boundary_mode))
    ax[1, 1].set_title("Watershed, {}, {}s".format(len(np.unique(segments_watershed)), watershed_time))

    # Debug: save fz and slic images
    #misc.imsave('out_fz.jpg', mark_boundaries(img, segments_fz, mode=boundary_mode))
    #misc.imsave('out_slic.jpg', mark_boundaries(img, segments_slic, mode=boundary_mode))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
