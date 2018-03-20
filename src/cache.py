import mmap
import os
import pickle
import re

from PIL import Image
from io import BytesIO
from enum import Enum
from multiprocessing import Lock
from logger import Logger

_MEMORY_MAPPED_IMAGE_CACHE_WRITE_LOCK = Lock()


class MemoryMapUpdateMode(Enum):
    UPDATE_ON_EVERY_WRITE = 0
    MANUAL = 1


class MemoryMappedImageCache(object):
    # A memory mapped image cache to reduce IO open/close operations.
    # Internally keeps an index of filename -> (start_byte, end_byte)
    # and stores the image data to a memory mapped file.

    def __init__(self, cache_path, max_mmap_file_size=0, read_only=False, memory_map_update_mode=MemoryMapUpdateMode.UPDATE_ON_EVERY_WRITE, write_to_secondary_file_cache=False):
        # type: (str, int, bool, MemoryMapUpdateMode) -> None

        self.cache_path = cache_path
        self.read_only = read_only
        self.max_mmap_file_size = long(max_mmap_file_size)
        self.memory_map_update_mode = memory_map_update_mode
        self.file_mode = "rb" if read_only or write_to_secondary_file_cache else "ab+"
        self.data_fp = None
        self.data_mm_fp = None
        self.index = None

        # Secondary file cache information
        self.write_to_secondary_file_cache = write_to_secondary_file_cache
        self.secondary_file_cache_path = os.path.join(os.path.dirname(self.cache_path), 'file_cache/')
        self.secondary_file_cache_index = set()

        # Primary memory mapped cache
        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path))

        # If we are creating a new cache
        if not os.path.exists(self.data_file_path) or not os.path.exists(self.index_file_path):
            self.index = dict()
            self.data_fp = self.open_file()
            self.data_fp.write('0xDEADBEEF')
            self.data_fp.flush()
        # If we are loading an existing cache
        else:
            # Open the memory mapped file and load the index file
            self.data_mm_fp = self.open_mmap_file()
            self.index = self.load_index_file()

        if not os.path.exists(os.path.dirname(self.secondary_file_cache_path)) and self.write_to_secondary_file_cache:
            os.makedirs(os.path.dirname(self.secondary_file_cache_path))
        else:
            self.update_secondary_file_cache_index()

    def __contains__(self, key):
        if self.index is not None and key in self.index:
            return True

        if self.secondary_file_cache_index is not None and key in self.secondary_file_cache_index:
            return True

        return False

    @property
    def data_file_path(self):
        return os.path.join(self.cache_path, "data.bin")

    @property
    def index_file_path(self):
        return os.path.join(self.cache_path, "index.pkl")

    @property
    def size(self):
        if self.index is not None:
            return len(self.index)
        return 0

    def keys(self):
        # type: () -> []
        if self.index is not None:
            return self.index.keys()
        return []

    def open_file(self):
        if self.data_fp is not None:
            self.data_fp.close()
            self.data_fp = None

        self.data_fp = open(self.data_file_path, self.file_mode)
        return self.data_fp

    def update_fp(self):
        self.open_file()

    def close_file(self):
        if self.data_fp is not None:
            self.data_fp.close()
            self.data_fp = None

    def open_mmap_file(self):
        if self.data_mm_fp is not None:
            self.data_mm_fp.close()
            self.data_mm_fp = None

        if self.data_fp is None:
            self.data_fp = self.open_file()

        try:
            self.data_fp.flush()
            if self.read_only or self.write_to_secondary_file_cache:
                self.data_mm_fp = mmap.mmap(self.data_fp.fileno(), length=self.max_mmap_file_size, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
            else:
                self.data_mm_fp = mmap.mmap(self.data_fp.fileno(), length=self.max_mmap_file_size, flags=mmap.MAP_SHARED)
        except IOError as e:
            Logger.instance().warn('Could not open memory mapped data file: {}'.format(e.message))

        return self.data_mm_fp

    def update_mmap_fp(self):
        self.open_mmap_file()

    def close_mmap_file(self):
        if self.data_mm_fp is not None:
            self.data_mm_fp.close()
            self.data_mm_fp = None

    def update_fps(self):
        self.update_fp()
        self.update_mmap_fp()

    def load_index_file(self):
        with open(self.index_file_path, 'rb') as f:
            return pickle.load(f)

    def save_index_file(self):
        with open(self.index_file_path, 'wb') as f:
            pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)

    def save(self):
        if self.data_mm_fp is not None:
            self.data_mm_fp.flush()

        if self.data_fp is not None:
            self.data_fp.flush()

        self.save_index_file()

    def close(self):
        if self.data_mm_fp is not None:
            self.data_mm_fp.close()
            self.data_mm_fp = None

        if self.data_fp is not None:
            self.data_fp.close()

    def update_secondary_file_cache_index(self):
        ext = 'jpg|jpeg|bmp|png'
        self.secondary_file_cache_index = set([f for root, _, files in os.walk(self.secondary_file_cache_path) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    def key_in_cache(self, key):
        key_in_primary_index = False
        key_in_secondary_index = False

        if self.index is not None:
            key_in_primary_index = key in self.index

        if self.secondary_file_cache_index is not None:
            key_in_secondary_index = key in self.secondary_file_cache_index

        return key_in_primary_index or key_in_secondary_index

    def set_image_to_cache(self, key, img, save_format=None):
        # type: (str, Image) -> None

        if self.read_only:
            raise ValueError('Cannot write to read_only cache')

        # Read to byte buffer because PIL closes the file pointer after saving
        img_bytes = BytesIO()

        if save_format is None:
            format = img.format
        else:
            format = save_format

        if self.write_to_secondary_file_cache:
            try:
                cached_img_path = os.path.join(self.secondary_file_cache_path, key)
                tmp_cached_img_path = cached_img_path + '.tmp'

                # If there is no tmp file and no real cached file - save the image
                if not os.path.exists(tmp_cached_img_path) and not os.path.exists(cached_img_path):
                    # Save is a long process and during save the file is not always valid,
                    # use .tmp extension and remove .tmp extension when save is complete
                    img.save(tmp_cached_img_path, format=save_format)
                    os.rename(tmp_cached_img_path, cached_img_path)

                self.secondary_file_cache_index.add(key)
            except Exception as e:
                Logger.instance().warn('Failed to write to secondary file cache: {}'.format(e.message))
                self.secondary_file_cache_index.remove(key)
        else:
            img.save(img_bytes, format=format)
            num_bytes = img_bytes.tell()
            img_bytes.seek(0)

            with _MEMORY_MAPPED_IMAGE_CACHE_WRITE_LOCK:
                self.data_fp.seek(0, os.SEEK_END)
                first_byte = self.data_fp.tell()
                last_byte = first_byte + num_bytes
                self.data_fp.write(img_bytes.read())
                self.index[key] = (first_byte, last_byte)

                if self.memory_map_update_mode == MemoryMapUpdateMode.UPDATE_ON_EVERY_WRITE:
                    self.update_mmap_fp()

    def get_image_from_cache(self, key, grayscale=False, load_to_memory=False):
        img = None

        # If the image is in cache
        if self.index is not None and key in self.index:
            try:
                bytes = self.index[key]
                #num_bytes = bytes[1] - bytes[0]
                #self.data_mm_fp.seek(bytes[0])
                #img = Image.open(BytesIO(self.data_mm_fp.read(num_bytes)))
                img = Image.open(BytesIO(self.data_mm_fp[bytes[0]:bytes[1]]))

                # Fix the filename of the PIL Image to match the key when reading from binary blob,
                # otherwise the filename will be empty/None
                img.filename = key
            except IOError as e:
                Logger.instance().warn('Failed to read image: {} from memory mapped file, error: {}'.format(key, e.message))
                return None
        # If the image is in secondary file cache
        elif self.secondary_file_cache_index is not None and key in self.secondary_file_cache_index:
            try:
                img = Image.open(os.path.join(self.secondary_file_cache_path, key))
            except IOError as e:
                Logger.instance().warn('Failed to read image: from secondary file cache, error: {}'.format(key, e.message))
                os.remove(os.path.join(self.secondary_file_cache_path, key))
                self.secondary_file_cache_index.remove(key)
                return None

        # If there is no image return none
        if img is None:
            return None

        # If we are supposed to load the image to memory
        if load_to_memory:
            img.load()

        # If this is a grayscale image and the mode is not 'L' -> convert
        if grayscale and img.mode != 'L':
            img = img.convert('L')

        return img
