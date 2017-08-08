# coding=utf-8

import tarfile
import os
import random
import re
import threading

from enum import Enum
from PIL import Image
from tarfile import TarInfo, TarFile

from abc import ABCMeta, abstractmethod, abstractproperty


##############################################
# IMAGE FILE
##############################################

class ImageFileType(Enum):
    NONE = -1
    TAR = 0,
    FILE_PATH = 1


class ImageFile(object):

    def __init__(self, file_path=None, tar_file=None, tar_info=None, tar_read_lock=None):
        # type: (str, TarFile, TarInfo) -> None

        """
        Creates an ImageFile instance either from the file path or from the tar
        arguments. You must provide either file_path or tar_file and tar_info, but
        not both.

        # Arguments
            :param file_path: File path to the image file
            :param tar_file: The TarFile where the image is if it's a member of a tar package
            :param tar_info: The TarInfo of the image
            :param tar_read_lock: the lock to acquire when reading the image from the tar package
        # Returns
            Nothing
        """

        if file_path is None and (tar_file is None or tar_info is None):
            raise ValueError('You must provide either image_set and tar_info or file_path')

        if file_path is not None and (tar_file is not None or tar_info is not None):
            raise ValueError('You cannot provide both file path and tar information')

        self._file_path = file_path
        self._tar_file = tar_file
        self._tar_info = tar_info
        self._tar_read_lock = tar_read_lock

        self._file_name = None
        self.type = ImageFileType.NONE

        if tar_file is not None and tar_info is not None:
            self._file_name = os.path.basename(tar_info.name)
            self.type = ImageFileType.TAR
        elif file_path is not None:
            self._file_name = os.path.basename(file_path)
            self.type = ImageFileType.FILE_PATH

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
    def file_name(self):
        return self._file_name

    def get_image(self, color_channels=3, target_size=None):
        # type: (int, tuple[int]) -> Image

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

        # Loading the data from the TAR file is not thread safe. So for each
        # image load the image data before releasing the lock. Regular files
        # can be lazy loaded, opening is enough.
        if self.type == ImageFileType.TAR:
            with self._tar_read_lock:
                f = self._tar_file.extractfile(self._tar_info)
                img = Image.open(f)
                img.load()
        elif self.type == ImageFileType.FILE_PATH:
            img = Image.open(self._file_path)
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

        if target_size:
            hw_tuple = (target_size[1], target_size[0])
            if img.size != hw_tuple:
                img = img.resize(hw_tuple)

        return img


##############################################
# IMAGE SET
##############################################

class ImageSet(object):

    def __init__(self, name, path_to_archive, file_list=None, mode='r'):
        # type: (str, str, list[str], str) -> None

        self.name = name
        self.path_to_archive = path_to_archive
        self._file_list = file_list
        self.mode = mode
        self._image_files = []
        self._file_name_to_image_file = dict()
        self._file_name_to_image_file_no_ext = dict()
        self._tar_file = None
        self._tar_read_lock = threading.Lock()

        if os.path.isfile(path_to_archive) and tarfile.is_tarfile(path_to_archive):
            self._tar_file = tarfile.open(name=path_to_archive, mode=mode)

            if self._tar_file is None:
                raise ValueError('Could not open tar archive from path: {}'.format(path_to_archive))

            # Filter out non files and hidden files
            tar_file_members = self._tar_file.getmembers()

            for tar_info in tar_file_members:
                if tar_info.isfile() and not os.path.basename(tar_info.name).startswith('.'):
                    img_file = ImageFile(tar_file=self._tar_file, tar_info=tar_info, tar_read_lock=self._tar_read_lock)
                    self._image_files.append(img_file)
                    file_name = os.path.basename(tar_info.name)
                    file_name_no_ext = os.path.splitext(file_name)[0]
                    self._file_name_to_image_file[file_name] = img_file
                    self._file_name_to_image_file_no_ext[file_name_no_ext] = img_file
        elif os.path.isdir(path_to_archive):
            image_paths = ImageSet.list_pictures(path_to_archive)

            for image_path in image_paths:
                if os.path.isfile(image_path) and not os.path.basename(image_path).startswith('.'):
                    img_file = ImageFile(file_path=image_path)
                    self._image_files.append(img_file)
                    file_name = os.path.basename(image_path)
                    file_name_no_ext = os.path.splitext(file_name)[0]
                    self._file_name_to_image_file[file_name] = img_file
                    self._file_name_to_image_file_no_ext[file_name_no_ext] = img_file
        else:
            raise ValueError('The given archive path is not recognized as a tar file or a directory: {}'.format(path_to_archive))

        # If a file list was provided filter so that image files only contains those files
        if self._file_list is not None:
            # Accelerate lookups by building a temporary set
            file_list_set = set(self._file_list)
            self._image_files = [img_file for img_file in self._image_files if img_file.file_name in file_list_set]

            # Check that the ImageFiles match to the given file set are identical if not raise an exception
            if len(self._image_files) != len(self._file_list):

                image_file_names = set([f.file_name for f in self._image_files])
                diff = file_list_set.difference(image_file_names)

                raise ValueError('Could not satisfy the given file list, image files and file list do not match: {} vs {}. Diff: {}'
                                 .format(len(self._image_files), len(self._file_list), diff))


    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
        return [os.path.join(root, f)
                for root, _, files in os.walk(directory) for f in files
                if re.match(r'([\w]+\.(?:' + ext + '))', f)]

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
        ret = self._file_name_to_image_file.get(file_name)

        if ret is None:
            file_name_no_ext = os.path.splitext(file_name)[0]
            ret = self._file_name_to_image_file_no_ext.get(file_name_no_ext)

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
    def sort(self):
        raise NotImplementedError('sort is not implemented in abstract DataSet')

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError('shuffle is not implemented in abstract DataSet')

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

    def sort(self):
        self._photo_image_set.sort()
        self._mask_image_set.sort()

    def shuffle(self):
        # Make sure the image sets are ordered in the same way
        self._photo_image_set.sort()
        self._mask_image_set.sort()

        # Shuffle two lists in the same way
        photos = self._photo_image_set.image_files
        masks = self._mask_image_set.image_files

        shuffled = zip(photos, masks)
        random.shuffle(shuffled)
        photos, masks = zip(*shuffled)
        photos = list(photos)
        masks = list(masks)

        if len(photos) != self._photo_image_set.size or len(masks) != self._mask_image_set.size:
            raise ValueError('Sizes have changed during shuffle, photos: {} vs {}, masks: {} vs {}'
                             .format(len(photos), self._photo_image_set.size, len(masks), self._mask_image_set.size))

        # Assign back
        self._photo_image_set.image_files = list(photos)
        self._mask_image_set.image_files = list(masks)

    def get_index(self, idx):
        return self._photo_image_set.image_files[idx], self._mask_image_set.image_files[idx]

    def get_indices(self, index_array):
        photos = [self._photo_image_set.image_files[i] for i in index_array]
        masks = [self._mask_image_set.image_files[i] for i in index_array]
        return zip(photos, masks)

    def get_files(self, file_names):
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
        material_samples = self.get_material_samples(index_array)
        file_names = [ms.file_name for ms in material_samples]
        photo_mask_pairs = self.get_files(file_names)

        return photo_mask_pairs, material_samples

    def get_material_samples(self, index_array):
        ret = []

        for i in range(len(index_array)):
            material_category_index = index_array[i][0]
            material_sample_index = index_array[i][1]
            ret.append(self.material_samples[material_category_index][material_sample_index])

        return ret

    def get_range(self, start, end):
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

    @property
    def photo_image_set(self):
        return self._photo_image_set

    @property
    def size(self):
        return self._photo_image_set.size

    def sort(self):
        self._photo_image_set.sort()

    def shuffle(self):
        random.shuffle(self._photo_image_set.image_files)

    def get_index(self, idx):
        return self._photo_image_set.image_files[idx]

    def get_indices(self, index_array):
        return [self._photo_image_set.image_files[i] for i in index_array]

    def get_range(self, start, end):
        return self._photo_image_set.image_files[start:end]
