import argparse
import os

from io import BytesIO
from diskcache import Cache
from PIL import Image

from ..utils import image_utils


def main():
    ap = argparse.ArgumentParser(description="Converts a folder of images into diskcache")
    ap.add_argument("-i", "--input", type=str, required=True, help="Path to images folder")
    ap.add_argument("-o", "--cache", type=str, required=True, help="Path to output cache folder")
    ap.add_argument("--eviction-policy", type=str, required=False, default="least-frequently-used", choices=["least-recently-stored", "least-recently-used", "least-frequently-used"])
    ap.add_argument("--cache-size-limit", type=int, required=False, default=100, help="Cache size limit in GBs")
    ap.add_argument("--cull-limit", type=int, required=False, default=10, help="The maximum number of keys to cull when adding a new item. Set to zero to disable automatic culling.")
    args = vars(ap.parse_args())

    input_path = args['input']
    cache_dir = args['cache']
    eviction_policy = args['eviction-policy']
    cache_size_limit = args['cache-size-limit']
    cull_limit = args['cull-limit']

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

    for img_path in img_paths:
        pil_img = Image.open(img_path)
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

        print '{}/{} - Cached: {}'.format(num_cached+1, num_images, key)
        num_cached += 1

    cache.close()
    print 'Done'


if __name__ == '__main__':
    main()
