import os
import argparse

from joblib import Parallel, delayed

from PIL import Image
from PIL.Image import Image as PILImage

import numpy as np

from ..utils import image_utils
from ..utils import general_utils
from ..utils.image_utils import ImageInterpolationType
from ..enums import ImageType


def pil_get_target_resize_shape_for_image(img, resize_shape):
    # type: (PILImage, tuple) -> tuple

    # If the resize shape is None just return the original image
    if resize_shape is None:
        return (img.height, img.width)

    assert (isinstance(resize_shape, list) or isinstance(resize_shape, tuple) or isinstance(resize_shape, np.ndarray))
    assert (len(resize_shape) == 2)
    assert (not (resize_shape[0] is None and resize_shape[1] is None))

    # Scale to match the sdim found in index 0
    if resize_shape[0] is not None and resize_shape[1] is None:
        target_sdim = resize_shape[0]
        img_sdim = min(img.size)

        if target_sdim == img_sdim:
            return (img.height, img.width)

        scale_factor = float(target_sdim) / float(img_sdim)
        target_shape = (int(round(img.height * scale_factor)), int(round(img.width * scale_factor)))
    # Scale the match the bdim found in index 1
    elif resize_shape[0] is None and resize_shape[1] is not None:
        target_bdim = resize_shape[1]
        img_bdim = max(img.size)

        if target_bdim == img_bdim:
            return (img.height, img.width)

        scale_factor = float(target_bdim) / float(img_bdim)
        target_shape = (int(round(img.height * scale_factor)), int(round(img.width * scale_factor)))
    # Scale to the exact shape
    else:
        target_shape = tuple(resize_shape)

        if target_shape[0] == img.height and target_shape[1] == img.width:
            return (img.height, img.width)

    return target_shape


def process_image(image_path, image_type, cache_path, resize_shapes, cval, interpolation_type):

    for resize_shape in resize_shapes:
        if resize_shape is None:
            continue

        img = Image.open(image_path)
        target_shape = pil_get_target_resize_shape_for_image(img=img, resize_shape=resize_shape)

        if target_shape[0] == img.height and target_shape[1] == img.width:
            continue

        # Cached file name is: <file_name>_<height>_<width>_<interp>_<img_type><file_ext>
        cached_img_name = os.path.splitext(os.path.basename(img.filename))
        filename_no_ext = cached_img_name[0]
        file_ext = cached_img_name[1]
        cached_img_name = '{}_{}_{}_{}_{}{}'.format(filename_no_ext,
                                                    target_shape[0],
                                                    target_shape[1],
                                                    interpolation_type.value,
                                                    image_type.value,
                                                    file_ext)

        # If there was no cached file - resize using PIL and cache
        # Use the same save format as the original file if it is given
        save_format = img.format

        if save_format is None:
            # Try determining format from file ending
            if file_ext.lower() == '.jpg' or file_ext.lower() == '.jpeg':
                save_format = 'JPEG'
            elif file_ext.lower() == '.png':
                save_format = 'PNG'
            else:
                # If we couldn't determine format from extension use the img type to guess
                save_format = 'PNG' if image_type.MASK else 'JPEG'

        cached_img_path = os.path.join(cache_path, cached_img_name)

        # Skip existing
        if os.path.exists(cached_img_path):
            return

        resized_img = image_utils.pil_resize_image_with_padding(img, shape=target_shape, cval=cval, interp=interpolation_type)
        resized_img.save(cached_img_path, format=save_format)
        print 'Saving image to: {}'.format(cached_img_path)


def main():
    ap = argparse.ArgumentParser(description="Creates a folder of resized images to function as a seed for resize cache")
    ap.add_argument("-i", "--input", type=str, required=True, help="Path to photos or masks folder")
    ap.add_argument("-o", "--cache", type=str, required=True, help="Path to output cache folder")
    ap.add_argument("-t", "--type", type=str, required=True, choices=["photo", "mask"], help="Type of the input images")
    ap.add_argument("-v", "--cval", type=str, required=True, help="Fill value in: R,G,B [0, 255] in case of necessary padding")
    ap.add_argument("--interpolation", type=str, required=True, choices=["nearest", "bilinear", "bicubic"])
    ap.add_argument("-j", "--jobs", type=int, required=True, help="Number of jobs (threads/processes)")
    ap.add_argument("-b", "--backend", type=str, required=True, choices=["multiprocessing", "threading"], help="Parallelization backend")
    args = vars(ap.parse_args())

    resize_shapes = [[1408, None], [512, None]]
    image_type_str_to_image_type = {"photo": ImageType.PHOTO, "mask": ImageType.MASK}
    interpolation_str_to_interpolation_type = {"nearest": ImageInterpolationType.NEAREST, "bilinear": ImageInterpolationType.BILINEAR, "bicubic": ImageInterpolationType.BICUBIC}

    input_path = args["input"]
    cache_path = args["cache"]
    image_type = image_type_str_to_image_type[args["type"]]
    interpolation_type = interpolation_str_to_interpolation_type[args["interpolation"]]
    cval = tuple([int(v.strip()) for v in args["cval"].split(',')])

    if len(cval) != 3:
        raise ValueError('Invalid cval: {}'.format(cval))

    n_jobs = args["jobs"]
    backend = args["backend"]

    print 'Reading images from: {}'.format(input_path)
    images = image_utils.list_pictures(input_path)
    print 'Found {} images'.format(len(images))

    general_utils.create_path_if_not_existing(cache_path)

    print 'Starting cache creation process with cache path: {}, image type: {}, cval: {}, interpolation: {}, jobs: {}, backend: {}'\
        .format(cache_path, image_type, cval, interpolation_type, n_jobs, backend)

    Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(process_image)(
            image_path,
            image_type,
            cache_path,
            resize_shapes,
            cval,
            interpolation_type) for image_path in images)


if __name__ == '__main__':
    main()
