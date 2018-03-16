import argparse
import os
import time

from PIL import Image, ImageFile

from ..utils import image_utils
from src.cache import MemoryMappedImageCache, MemoryMapUpdateMode


def main():
    ap = argparse.ArgumentParser(description="Converts a folder of images into diskcache")
    ap.add_argument("-i", "--input", type=str, required=True, help="Path to images folder")
    ap.add_argument("-o", "--cache", type=str, required=True, help="Path to output cache folder")
    args = vars(ap.parse_args())

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    input_path = args['input']
    cache_dir = args['cache']

    print 'Reading images from: {}'.format(input_path)
    img_paths = image_utils.list_pictures(input_path)
    print 'Found {} images'.format(len(img_paths))

    # Create the cache
    print 'Creating cache: {}'.format(cache_dir)
    cache = MemoryMappedImageCache(cache_path=cache_dir, memory_map_update_mode=MemoryMapUpdateMode.MANUAL)

    num_images = len(img_paths)
    num_cached = 0
    start_time = time.time()
    num_failed_images = 0
    print_every = 1000

    for img_path in img_paths:

        key = os.path.basename(img_path)

        if key not in cache:
            try:
                pil_img = Image.open(img_path)
            except IOError as e:
                num_failed_images += 1
                print 'WARNING: Failed to read image {}: {}'.format(img_path, e.message)
                continue

            key = os.path.basename(pil_img.filename)
            cache.set_image_to_cache(key, pil_img)

        num_cached += 1
        etr = (num_images - num_cached) * ((time.time()-start_time) / num_cached)

        if num_cached % print_every == 0:
            print '{}/{} - ETA: {:.4f} - Cached: {}'.format(num_cached, num_images, etr, key)

    print 'Saving cache'
    cache.save()
    print 'Cache size: {}'.format(cache.size)
    cache.close()
    print 'Cache conversion finished with {} cached images, {} failed images'.format(num_cached, num_failed_images)


if __name__ == '__main__':
    main()
