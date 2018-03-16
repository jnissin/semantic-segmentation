# coding=utf-8

import tarfile
import os
import random
import re
import multiprocessing

from enum import Enum
from PIL import Image
from tarfile import TarInfo, TarFile
from cache import MemoryMappedImageCache, MemoryMapUpdateMode
from io import BytesIO

from abc import ABCMeta, abstractmethod, abstractproperty


##############################################
# IMAGE FILE
##############################################

class ImageFileType(Enum):
    NONE = -1,
    TAR = 0,
    FILE_PATH = 1,
    MMI_CACHE = 2


class ImageFile(object):

    def __init__(self, image_path=None, tar_info=None, image_key=None, shared_resources=None):
        # type: (str, TarInfo, str, ImageSetSharedResources) -> None

        """
        Creates an ImageFile instance either from the file path or from the tar
        arguments. You must provide either file_path or tar_info. With tar_info
        you should always provide shared resources, with image_path it's optional, but if
        not provided image_path is assumed to be absolute path. Otherwise, the image_path
        is combined with path_to_archive of shared resources.

        # Arguments
            :param image_path: File path to the image file
            :param tar_info: The TarInfo of the image
            :param image_key: The key of the image in the MMI
            :param shared_resources: Shared resources in the image set
        # Returns
            Nothing
        """
        if image_path is None and tar_info is None and image_key is None:
            raise ValueError('You must provide either an image_path or shared_resources and tar_info/image_key')

        if (image_path is not None and tar_info is not None) or \
           (image_path is not None and image_key is not None) or \
           (image_key is not None and tar_info is not None):
            raise ValueError('Please provide only one of the following: image_path/tar_info/image_key')

        self._image_path = image_path
        self._shared_resources = shared_resources
        self._tar_info = tar_info
        self._image_key = image_key
        self.type = ImageFileType.NONE

        # Validate the image path if not in a tar archive - better fail early
        if self._image_path is not None:
            if self._shared_resources is None:
                if not os.path.exists(self._image_path):
                    raise ValueError('Image path {} does not exist'.format(self._image_path))
            else:
                if not os.path.exists(os.path.join(self._shared_resources.path_to_archive, self._image_path)):
                    raise ValueError('Image path {} does not exist'.format(os.path.join(self._shared_resources.path_to_archive, self._image_path)))

        if image_path is not None:
            self.type = ImageFileType.FILE_PATH
        elif self.mmi_cache is not None and image_key is not None:
            self.type = ImageFileType.MMI_CACHE
        elif self.tar_file is not None and tar_info is not None:
            self.type = ImageFileType.TAR

    def __eq__(self, other):
        if not isinstance(other, ImageFile):
            raise NotImplementedError('Comparison between other than ImageMember objects has not been implemented')

        return self.file_name.lower() == other.file_name.lower()

    def __ne__(self, other):
        if not isinstance(other, ImageFile):
            raise NotImplementedError('Comparison between other than ImageMember objects has not been implemented')

        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, ImageFile):
            raise NotImplementedError('Comparison between other than ImageMember objects has not been implemented')

        return self.file_name < other.file_name

    def __gt__(self, other):
        if not isinstance(other, ImageFile):
            raise NotImplementedError('Comparison between other than ImageMember objects has not been implemented')

        return self.file_name > other.file_name

    @property
    def file_path(self):
        # Combine the image path with the shared resource path to archive
        if self._shared_resources is not None and self.image_path is not None:
            # The shared resource should not have a tar file in this case
            if self._shared_resources.path_to_archive is not None and self._shared_resources.tar_file is None:
                return os.path.join(self._shared_resources.path_to_archive, self.image_path)
        # Assume the image path is the absolute path
        elif self.image_path is not None:
            return self.image_path

        return None

    @property
    def image_path(self):
        return self._image_path

    @property
    def tar_info(self):
        return self.tar_info

    @property
    def tar_file(self):
        if self._shared_resources is not None:
            return self._shared_resources.tar_file
        return None

    @property
    def mmi_cache(self):
        if self._shared_resources is not None:
            if self._shared_resources.mmi_cache is None:
                self.reopen_mmi_cache_handles()

        return self._shared_resources.mmi_cache

    @property
    def tar_read_lock(self):
        if self._shared_resources is not None:
            return self._shared_resources.tar_read_lock
        return None

    @property
    def file_name(self):
        if self._image_path is not None:
            return os.path.basename(self.image_path)
        elif self._tar_info is not None:
            return os.path.basename(self.tar_info.name)
        elif self._image_key is not None:
            return os.path.basename(self._image_key)

    def reopen_mmi_cache_handles(self):
        if self._shared_resources is None:
            raise ValueError('No shared resources available - cannot open mmi cache')

        if self._shared_resources.mmi_cache_path is None:
            raise ValueError('mmi_cache_path of shared resources is none - cannot open mmi cache handle')

        self._shared_resources.reopen_mmi_cache()

    def get_image(self, color_channels=3, target_size=None):
        # type: (int, tuple[int, int]) -> Image

        """
        Returns a PIL image.

        # Arguments
            :param color_channels: number of color channels in the image (1,3 or 4)
            :param target_size: target size to resize the image to
        # Returns
            :return: a PIL image
        """

        if color_channels != 1 and color_channels != 3 and color_channels != 4:
            raise ValueError('Number of channels must be 1, 3 or 4')

        img = None

        if self.type == ImageFileType.FILE_PATH:
            img = Image.open(self.file_path)
        elif self.type == ImageFileType.MMI_CACHE:
            try:
                img = Image.open(BytesIO(self.mmi_cache[self._image_key]))
            except Exception as e:
                # Attempt to reopen the MMI cache and read again
                self.reopen_mmi_cache_handles()
                img = Image.open(BytesIO(self.mmi_cache[self._image_key]))
        # Loading the data from the TAR file is not thread safe. So for each
        # image load the image data before releasing the lock. Regular files
        # can be lazy loaded, opening is enough.
        elif self.type == ImageFileType.TAR:
            with self.tar_read_lock:
                f = self.tar_file.extractfile(self.tar_info)
                img = Image.open(f)
                img.load()
        else:
            raise ValueError('Cannot open ImageFileType: {}'.format(self.type))

        if color_channels == 1:
            if img.mode != 'L':
                img = img.convert('L')
        elif color_channels == 3:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        elif color_channels == 4:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

        if target_size is not None:
            hw_tuple = (target_size[1], target_size[0])
            if img.size != hw_tuple:
                img = img.resize(hw_tuple)

        return img


##############################################
# IMAGE SET
##############################################

class ImageSetSharedResources(object):

    def __init__(self, path_to_archive, tar_file=None, tar_read_lock=None, mmi_cache=None, mmi_cache_path=None):
        self._path_to_archive = path_to_archive
        self._tar_read_lock = tar_read_lock
        self._tar_file = tar_file
        self._mmi_cache = mmi_cache
        self._mmi_cache_path = mmi_cache_path

    @property
    def path_to_archive(self):
        return self._path_to_archive

    @property
    def tar_file(self):
        return self._tar_file

    @property
    def tar_read_lock(self):
        return self._tar_read_lock

    @property
    def mmi_cache(self):
        # type: () -> MemoryMappedImageCache
        return self._mmi_cache

    @property
    def mmi_cache_path(self):
        # type: () -> str
        return self._mmi_cache_path

    def reopen_mmi_cache(self):
        if self.mmi_cache is not None:
            self.mmi_cache.update_fps()
        else:
            self._mmi_cache = MemoryMappedImageCache(self.mmi_cache_path, read_only=True, memory_map_update_mode=MemoryMapUpdateMode.MANUAL)


class ImageSet(object):

    def __init__(self, name, path_to_archive, file_list=None, mode='r'):
        # type: (str, str, list[str], str) -> None

        self.name = name
        self.path_to_archive = path_to_archive
        self.mode = mode
        self._image_files = []
        self._file_name_to_image_file = dict()
        self._image_set_shared_resources = None

        if os.path.isfile(path_to_archive) and tarfile.is_tarfile(path_to_archive):
            # Instantiate the tar file and a tar file read lock (doesn't support multi-threading)
            tar_file = tarfile.open(name=path_to_archive, mode=mode)
            tar_read_lock = multiprocessing.Lock()

            if tar_file is None:
                raise ValueError('Could not open tar archive from path: {}'.format(path_to_archive))

            # Create the shared resources
            self._image_set_shared_resources = ImageSetSharedResources(path_to_archive=None, tar_file=tar_file, tar_read_lock=tar_read_lock)

            # Filter out non files and hidden files
            tar_file_members = tar_file.getmembers()

            for tar_info in tar_file_members:
                if tar_info.isfile() and not os.path.basename(tar_info.name).startswith('.'):
                    img_file = ImageFile(image_path=None, tar_info=tar_info, shared_resources=self._image_set_shared_resources)
                    self._image_files.append(img_file)
                    file_name = os.path.basename(tar_info.name)
                    self._file_name_to_image_file[file_name] = img_file
        elif os.path.isdir(path_to_archive):
            # If the dataset is a MemoryMappedImageCache
            if os.path.exists(os.path.join(path_to_archive, 'data.bin')) and os.path.exists(os.path.join(path_to_archive, 'index.pkl')):
                cache = MemoryMappedImageCache(path_to_archive, read_only=True, memory_map_update_mode=MemoryMapUpdateMode.MANUAL)
                self._image_set_shared_resources = ImageSetSharedResources(path_to_archive=None, tar_file=None, tar_read_lock=None, mmi_cache_path=path_to_archive, mmi_cache=cache)

                image_keys = cache.keys()

                for image_key in image_keys:
                    img_file = ImageFile(image_path=None, tar_info=None, image_key=image_key, shared_resources=self._image_set_shared_resources)
                    file_name = img_file.file_name
                    self._image_files.append(img_file)
                    self._file_name_to_image_file[file_name] = img_file
            else:
                image_paths = ImageSet.list_pictures(path_to_archive)

                # Remove the shared part of the path - also: filter hidden files and non-files
                image_paths = [os.path.relpath(p, start=self.path_to_archive) for p in image_paths if os.path.isfile(p) and not os.path.basename(p).startswith('.')]

                # Create the shared resources
                self._image_set_shared_resources = ImageSetSharedResources(path_to_archive=self.path_to_archive, tar_file=None, tar_read_lock=None)

                for image_path in image_paths:
                    img_file = ImageFile(image_path=image_path, tar_info=None, shared_resources=self._image_set_shared_resources)
                    file_name = img_file.file_name
                    self._image_files.append(img_file)
                    self._file_name_to_image_file[file_name] = img_file
        else:
            raise ValueError('The given archive path is not recognized as a tar file or a directory: {}'.format(path_to_archive))

        # If a file list was provided filter so that image files only contains those files
        if file_list is not None and len(file_list) > 0:
            # Accelerate lookups by building a temporary set
            file_list_set = set(file_list)
            filtered_image_files = []

            # Filter according to file list
            for img_file in self._image_files:
                file_name = img_file.file_name.lower()
                in_file_list = file_name in file_list_set

                if not in_file_list:
                    # Remove also from the file name mapping
                    del self._file_name_to_image_file[file_name]
                else:
                    filtered_image_files.append(img_file)

            self._image_files = filtered_image_files

            # Check that the ImageFiles match to the given file set i.e. they are identical - if not raise an exception
            if len(self._image_files) != len(file_list):

                image_file_names = set([f.file_name for f in self._image_files])

                if len(file_list_set) > len(image_file_names):
                    diff = list(image_file_names.difference(file_list_set))
                else:
                    diff = list(file_list_set.difference(image_file_names))

                raise ValueError('Could not satisfy the given file list, image files and file list do not match: {} vs {}. Diff (first 10): {}'
                                 .format(len(self._image_files), len(file_list), diff[0:min(len(diff), 10)]))

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
        return [os.path.join(root, f)
                for root, _, files in os.walk(directory) for f in files
                if ImageSet.is_image_file(f, ext=ext)]

    @staticmethod
    def is_image_file(f, ext='jpg|jpeg|bmp|png'):
        return re.match(r'([\w]+\.(?:' + ext + '))', f)

    @property
    def image_files(self):
        return self._image_files

    @image_files.setter
    def image_files(self, val):
        self._image_files = val

    @property
    def size(self):
        return len(self._image_files)

    def sort(self):
        self._image_files.sort()

    def get_image_file_by_file_name(self, file_name):
        # Attempt to find the file name
        ret = self._file_name_to_image_file.get(file_name)

        # If the ImageFile was not found
        if ret is None:
            # If the file name has an extension - try with the same name without the extension
            if ImageSet.is_image_file(file_name):
                file_name_no_ext = os.path.splitext(file_name)[0]
                ret = self._file_name_to_image_file.get(file_name_no_ext)
            # If the file name did not have an extension - try with the same file name but common image extensions
            else:
                exts = ['jpg', 'png', 'jpeg', 'bmp']
                file_name_no_ext = os.path.splitext(file_name)[0]
                for ext in exts:
                    ret = self._file_name_to_image_file.get('{}.{}'.format(file_name_no_ext, ext))
                    if ret is not None:
                        return ret

        return ret


##############################################
# IMAGE DATA SET
##############################################

class DataSet(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractproperty
    def size(self):
        raise NotImplementedError('size is not implemented in abstract DataSet')

    @abstractmethod
    def get_index(self, idx):
        raise NotImplementedError('get_index is not implemented in abstract DataSet')

    @abstractmethod
    def get_indices(self, index_array):
        raise NotImplementedError('get_indices is not implemented in abstract DataSet')

    @abstractmethod
    def get_range(self, start, end):
        raise NotImplementedError('get_range is not implemented in abstract DataSet')


##############################################
# LABELED IMAGE DATA SET
##############################################

class LabeledImageDataSet(DataSet):

    def __init__(self, name, path_to_photo_archive, path_to_mask_archive, photo_file_list=None, mask_file_list=None, material_samples=None):
        # type: (str, str, str, list[str], list[str]) -> None

        """
        Creates a labeled image dataset from the two parameter archives.
        The archives must be either paths to folders or tar.gz archives.

        # Arguments
            :param name: name for the dataset
            :param path_to_photo_archive: path to the photos (folder or tar.gz)
            :param path_to_mask_archive: path to the masks (folder or tar.gz)
            :param photo_file_list: list of photo files to use from the photo archive, all used if None
            :param mask_file_list: list of mask files to use from the mask archive, all used if None
            :param material_samples: a list of calculated material samples for the data set, defaults to None
        # Returns
            Nothing
        """
        super(LabeledImageDataSet, self).__init__(name=name)

        self._photo_image_set = ImageSet(self.name + '_photos', path_to_photo_archive, photo_file_list)
        self._mask_image_set = ImageSet(self.name + '_masks', path_to_mask_archive, mask_file_list)
        self._material_samples = material_samples

        # Make sure the photos and masks are organized in the same way
        self._photo_image_set.sort()
        self._mask_image_set.sort()

        if self._photo_image_set.size != self._mask_image_set.size:
            raise ValueError('Non-matching photo and mask data set sizes: {} vs {}'.format(self._photo_image_set.size, self._mask_image_set.size))

    @property
    def photo_image_set(self):
        return self._photo_image_set

    @property
    def mask_image_set(self):
        return self._mask_image_set

    @property
    def material_samples(self):
        return self._material_samples

    @property
    def size(self):
        return self._photo_image_set.size

    def get_index(self, idx):
        return self._photo_image_set.image_files[idx], self._mask_image_set.image_files[idx]

    def get_indices(self, index_array):
        photos = [self._photo_image_set.image_files[i] for i in index_array]
        masks = [self._mask_image_set.image_files[i] for i in index_array]
        return zip(photos, masks)

    def get_files(self, file_names):
        # type: (list[str]) -> list[tuple[ImageFile, ImageFile]]
        photos = []
        masks = []

        for file_name in file_names:
            photo = self._photo_image_set.get_image_file_by_file_name(file_name)

            if photo is None:
                raise ValueError('Could not find photo file from ImageSet with file name: {}'.format(file_name))

            mask = self._mask_image_set.get_image_file_by_file_name(file_name)

            if mask is None:
                raise ValueError('Could not find mask file from ImageSet with file name: {}'.format(file_name))

            photos.append(photo)
            masks.append(mask)

        return zip(photos, masks)

    def get_files_and_material_samples(self, index_array):
        # type: (list[tuple[int, int]]) -> (list[tuple[ImageFile, ImageFile]], list[MaterialSample])

        """
        Takes an index array describing the material samples as an argument. The indexing is expected to
        be (material_category_idx, material_sample_idx) where the latter is the sample idx of the material
        sample within the given material category.

        The function returns a list of ImageFile pairs (photo_file, mask_file)

        # Arguments
            :param index_array: The material sample index array.
        # Returns
            :return: a list of ImageFile pairs (photo_file, mask_file)
        """

        material_samples = self.get_material_samples(index_array)
        file_names = [ms.file_name_no_ext for ms in material_samples]
        photo_mask_pairs = self.get_files(file_names)

        return photo_mask_pairs, material_samples

    def get_material_samples(self, index_array):
        # type: (list[tuple[int, int]]) -> list[MaterialSample]

        msamples = []

        for i in range(len(index_array)):
            material_category_index = index_array[i][0]
            material_sample_index = index_array[i][1]
            msamples.append(self.material_samples[material_category_index][material_sample_index])

        return msamples

    def get_range(self, start, end):
        # type: (int, int) -> (tuple[list(ImageFile), list(ImageFile)])

        photos = self._photo_image_set.image_files[start:end]
        masks = self._mask_image_set.image_files[start:end]
        return zip(photos, masks)


##############################################
# UNLABELED IMAGE DATA SET
##############################################

class UnlabeledImageDataSet(DataSet):
    def __init__(self, name, path_to_photo_archive, photo_file_list=None):
        # type: (str, str, list[str]) -> None

        """
        Creates an unlabeled image dataset from the given archive. The archive can
        either be a folder path or a tar.gz archive.

        # Arguments
            :param name: name for the data set
            :param path_to_photo_archive: path to the archive (folder or tar.gz)
            :param photo_file_list: list of files to use from the archive, all used if None
        # Returns
            Nothing
        """

        super(UnlabeledImageDataSet, self).__init__(name=name)
        self._photo_image_set = ImageSet(self.name + '_photos', path_to_photo_archive, photo_file_list)
        self._photo_image_set.sort()

    @property
    def photo_image_set(self):
        return self._photo_image_set

    @property
    def size(self):
        return self._photo_image_set.size

    def get_index(self, idx):
        return self._photo_image_set.image_files[idx]

    def get_indices(self, index_array):
        return [self._photo_image_set.image_files[i] for i in index_array]

    def get_range(self, start, end):
        return self._photo_image_set.image_files[start:end]
