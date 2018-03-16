import mmap
import os
import pickle

from PIL import Image
from io import BytesIO
from enum import Enum
from multiprocessing import Lock

_MEMORY_MAPPED_IMAGE_CACHE_WRITE_LOCK = Lock()


class MemoryMapUpdateMode(Enum):
    UPDATE_ON_EVERY_WRITE = 0
    MANUAL = 1


class MemoryMappedImageCache(object):
    # A memory mapped image cache to reduce IO open/close operations.
    # Internally keeps an index of filename -> (start_byte, end_byte)
    # and stores the image data to a memory mapped file.

    def __init__(self, cache_path, max_mmap_file_size=0, read_only=False, memory_map_update_mode=MemoryMapUpdateMode.UPDATE_ON_EVERY_WRITE):
        # type: (str, int, bool, MemoryMapUpdateMode) -> None

        self.cache_path = cache_path
        self.read_only = read_only
        self.max_mmap_file_size = long(max_mmap_file_size)
        self.memory_map_update_mode = memory_map_update_mode
        self.file_mode = "rb" if read_only else "ab+"
        self.data_fp = None
        self.data_mm_fp = None
        self.index = None

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

    def __contains__(self, key):
        return key in self.index

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
            self.data_mm_fp = mmap.mmap(self.data_fp.fileno(), length=self.max_mmap_file_size)
        except IOError as e:
            print 'ERROR: Could not open memory mapped data file: {}'.format(e.message)

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

    def key_in_cache(self, key):
        if self.index is not None:
            return key in self.index
        return False

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

    def get_image_from_cache(self, key):
        # If the image is in cache
        if key in self.index:
            bytes = self.index[key]
            img = Image.open(BytesIO(self.data_mm_fp[bytes[0]:bytes[1]]))
            return img
        else:
            return None
