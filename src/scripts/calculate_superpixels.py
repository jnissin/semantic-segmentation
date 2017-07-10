# coding=utf-8

import argparse
import time
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import load_img, img_to_array


def get_superpixel_segmentation(np_img, n_segments, sigma=5, compactness=10.0, max_iter=10):
    # type: (np.array, int, int, float) -> np.array

    # Apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(np_img, n_segments=n_segments, sigma=sigma, compactness=compactness, max_iter=max_iter)
    return segments


def main():
    # Construct the argument parser and parge arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-n", "--nsegments", required=False, default=100, type=int, help="Number of segments")
    ap.add_argument("-s", "--sigma", required=False, default=5, type=int, help="Sigma value for SLIC")
    ap.add_argument("-c", "--compactness", required=False, default=10.0, type=float, help="Compactness value for SLIC")
    ap.add_argument("-m", "--maxiter", required=False, default=10, type=int,
                    help="Maximum number of iterations for SLIC")
    args = vars(ap.parse_args())

    image = load_img(args["image"])
    np_img = img_to_array(image)

    # Normalize the image to [-1, 1]
    np_img -= 128.0
    np_img /= 128.0

    n_segments = args["nsegments"]
    sigma = args["sigma"]
    compactness = args["compactness"]

    start_time = time.time()
    segments = get_superpixel_segmentation(np_img, n_segments, sigma, compactness)
    duration = time.time() - start_time
    print "SLIC superpixel calculation took: %.4f sec" % duration

    # Show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments %.4f sec" % (n_segments, duration))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    plt.show()

    print "Done"


if __name__ == "__main__":
    main()