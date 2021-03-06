import argparse
import os
import time

from io import BytesIO
from diskcache import Cache
from PIL import Image, ImageFile

from ..utils import image_utils


def main():
    ap = argparse.ArgumentParser(description="Converts a folder of images into diskcache")
    ap.add_argument("-i", "--input", type=str, required=True, help="Path to images folder")
    ap.add_argument("-o", "--cache", type=str, required=True, help="Path to output cache folder")
    ap.add_argument("--evictionpolicy", type=str, required=False, default="least-recently-stored", choices=["least-recently-stored", "least-recently-used", "least-frequently-used"])
    ap.add_argument("--cachesizelimit", type=int, required=False, default=100, help="Cache size limit in GBs")
    ap.add_argument("--culllimit", type=int, required=False, default=10, help="The maximum number of keys to cull when adding a new item. Set to zero to disable automatic culling.")
    args = vars(ap.parse_args())

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    input_path = args['input']
    cache_dir = args['cache']
    eviction_policy = args['evictionpolicy']
    cache_size_limit = args['cachesizelimit']
    cull_limit = args['culllimit']

    print 'Reading images from: {}'.format(input_path)
    img_paths = image_utils.list_pictures(input_path)
    print 'Found {} images'.format(len(img_paths))

    # Create the cache
    gb_in_bytes = 1073741824
    size_limit = gb_in_bytes * cache_size_limit
    eviction_policy = eviction_policy

    print 'Creating cache: {}, size_limit: {}, eviction_policy: {}, cull_limit: {}'.format(cache_dir, size_limit, eviction_policy, cull_limit)
    cache = Cache(cache_dir, size_limit=size_limit, eviction_policy=eviction_policy, cull_limit=cull_limit)

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

            img_bytes = BytesIO()
            pil_img.save(img_bytes, format=pil_img.format)
            img_bytes.seek(0)

            if pil_img.format == 'JPEG':
                tag = 'photo'
            elif pil_img.format == 'PNG':
                tag = 'mask'
            else:
                tag = None

            key = os.path.basename(pil_img.filename)
            cache.set(key, img_bytes, read=True, tag=tag)

        num_cached += 1
        etr = (num_images - num_cached) * ((time.time()-start_time) / num_cached)

        if num_cached % print_every == 0:
            print '{}/{} - ETA: {:.4f} - Cached: {}'.format(num_cached, num_images, etr, key)

    cache.close()
    print 'Cache conversion finished with {} cached images, {} failed images'.format(num_cached, num_failed_images)


if __name__ == '__main__':
    main()
