# coding=utf-8

import numpy as np
import os
import random
import ctypes
import multiprocessing

from multiprocessing import Array

from keras_extensions.utils.data_utils import Sequence

from abc import ABCMeta, abstractmethod
from enum import Enum

from logger import Logger
from enums import BatchDataFormat


_DATA_SET_ITERATOR_UUID = 0


def _get_next_data_set_iterator_uuid():
    global _DATA_SET_ITERATOR_UUID
    uuid = _DATA_SET_ITERATOR_UUID
    _DATA_SET_ITERATOR_UUID += 1
    return uuid


class ExtendedDictionary(dict):
    def __init__(self, default, **kwargs):
        self.default = default
        super(ExtendedDictionary, self).__init__(**kwargs)

    def __getitem__(self, key):
        if key in self:
            return super(ExtendedDictionary, self).__getitem__(key)
        return self.default


class BatchIndexBuffer(object):

    def __init__(self,
                 n,
                 batch_size,
                 shuffle,
                 seed,
                 logger,
                 initial_epoch,
                 num_queued_epochs=4):
        # type: (int, int, bool, int, Logger, int, int) -> None

        """
        # Arguments
            :param data_generator: DataGenerator, a data generator that returns batches of data when provided with batch indices
            :param n: integer, total number of samples in the data set to loop over
            :param batch_size: integer, number of samples in a batch
            :param shuffle: boolean, whether to shuffle the data between epochs.
            :param seed: random seeding for data shuffling.
            :param logger: logger instance for logging
            :param initial_epoch: initial epoch
            :param num_queued_epochs: number of epochs for which batch sample indices should be queued
        # Returns
            Nothing
        """
        self.n = n
        self.batch_size = min(batch_size, n)            # The batch size could in theory be bigger than the data set size
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger
        self.initial_epoch = initial_epoch
        self.num_queued_epochs = num_queued_epochs

        # Generate the initial data
        self.epoch_queue = ExtendedDictionary(default=np.arange(n) if not self.shuffle else None)

        # We only need to generate data if we shuffle between epochs
        # otherwise the data is just the default return value for the epoch queue
        if self.shuffle:
            for i in range(0, self.num_queued_epochs):
                e_idx = self.initial_epoch+i
                self.epoch_queue[e_idx] = self._create_index_array_for_epoch(e_idx=e_idx)

    def _get_index_array_for_epoch(self, e_idx):
        # type: (int) -> np.ndarray

        """
        Generates the batch sample indices for a specific epoch.

        # Arguments
            :param e_idx: epoch index
        # Returns
            :return: sample indices for the specified epoch
        """

        if e_idx not in self.epoch_queue and self.shuffle:
            self._update_epoch_queue(r_e_idx=e_idx)

        return self.epoch_queue[e_idx]

    def _create_index_array_for_epoch(self, e_idx):
        # type: (int) -> np.ndarray

        """
        Generates the batch sample indices for a specific epoch.

        # Arguments
            :param e_idx: epoch index
        # Returns
            :return: sample indices for the specified epoch
        """

        if self.shuffle:
            np.random.seed(self.seed + e_idx)
            index_array = np.random.permutation(self.n)
        else:
            index_array = np.arange(self.n)

        return index_array

    def _update_epoch_queue(self, r_e_idx):
        # type: (int, int) -> ()

        """
        Moves the epoch queue window ahead by half the number of queued epochs.
        For example if the queue had epochs 0,1,2,3 after this call the queue
        would have epochs 2,3,4,5.

        # Arguments
            :param r_e_idx: requested epoch index
        # Returns
            Nothing
        """
        keys = self.epoch_queue.keys()
        keys.sort()
        max_key = max(keys)
        min_key = min(keys)
        window_move_size = len(keys) / 2

        # Check whether there is a bug and we are trying to request an epoch from the past - warn but don't crash
        if r_e_idx < min_key:
            self.logger.warn('Requested a past epoch: min key: {}, requested epoch: {}'.format(min_key, r_e_idx))
            self.epoch_queue[r_e_idx] = self._create_index_array_for_epoch(e_idx=r_e_idx)
        # Otherwise move the queued epochs window by half forward
        else:
            # Remove the old first half of the queued epochs
            for i in range(0, window_move_size):
                k = keys[i]
                if k in self.epoch_queue:
                    del self.epoch_queue[k]

            # Add a new half in to the queued epochs
            for i in range(0, window_move_size):
                k = max_key+i+1
                self.epoch_queue[k] = self._create_index_array_for_epoch(e_idx=k)

        # Final sanity check
        if r_e_idx not in self.epoch_queue:
            self.logger.warn('Requested for an epoch ({}) outside the sliding window distance: {}'.format(r_e_idx, window_move_size))
            self.epoch_queue[r_e_idx] = self._create_index_array_for_epoch(e_idx=r_e_idx)

    def get_batch_indices(self, e_idx, b_idx):
        # type: (int, int) -> (np.ndarray)

        """
        Returns the sample indices for a given epoch and batch index as a numpy array.

        # Arguments
            :param e_idx: epoch index
            :param b_idx: batch index (within the epoch)
        # Returns
            :return: sample indices for the given epoch
        """

        # Batch indices
        index_array = self._get_index_array_for_epoch(e_idx=e_idx)

        current_sample_index = (b_idx * self.batch_size) % self.n

        if self.n > b_idx + self.batch_size:
            samples_in_batch = self.batch_size
            is_last_batch = False
        else:
            samples_in_batch = self.n - current_sample_index
            is_last_batch = True

        batch = index_array[current_sample_index:current_sample_index + samples_in_batch]

        return batch


class DataSetIterator(Sequence):
    """
    Abstract base class of data set iterator. This class supports multiprocess
    iteration with Keras by implementing the Sequence interface.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 data_generator,
                 n_labeled,
                 labeled_batch_size,
                 n_unlabeled,
                 unlabeled_batch_size,
                 shuffle,
                 seed,
                 logger,
                 initial_epoch):
        # type: (DataGenerator, int, int, int, int, bool, int, Logger, int) -> None

        """
        # Arguments
            :param data_generator: DataGenerator, a data generator that returns batches of data when provided with batch indices
            :param n_labeled: integer, total number of labeled samples in the data set to loop over
            :param labeled_batch_size: integer, size of labeled data in batch
            :param n_unlabeled: integer, total number of unlabeled samples in the data set to loop over
            :param unlabeled_batch_size: integer, size of unlabeled data in batch
            :param shuffle: boolean, whether to shuffle the data between epochs.
            :param seed: random seeding for data shuffling.
            :param logger: logger instance for logging
            :param initial_epoch: initial epoch
        # Returns
            Nothing
        """
        self.data_generator = data_generator                                    # DataGenerator instance
        self.n_labeled = n_labeled                                              # Size of the labeled data set (n samples)
        self.n_unlabeled = n_unlabeled                                          # Size of the unlabeled data set (n samples)
        self.labeled_batch_size = min(labeled_batch_size, n_labeled)            # The batch size could in theory be bigger than the data set size
        self.unlabeled_batch_size = min(unlabeled_batch_size, n_unlabeled)
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger
        self.initial_epoch = initial_epoch
        self._uuid = _get_next_data_set_iterator_uuid()

        # The following member variables only work when not used in multiprocessing context
        self.global_step_index = self.initial_epoch * self.num_steps_per_epoch  # The global step index (how many batches have been processed altogether)
        self._lock = None

        # Handle the initial random seeding
        np.random.seed(self.seed)
        random.seed(self.seed)

    def reset(self):
        self.global_step_index = self.initial_epoch * self.num_steps_per_epoch

    @property
    def uuid(self):
        # type: () -> int
        return self._uuid

    @property
    def epoch_index(self):
        # type: () -> int

        """
        The index of the epoch. Only works when using non-multiprocessing context.

        # Arguments
            None
        # Returns
            :return: index of the epoch
        """

        return self.global_step_index / self.num_steps_per_epoch

    @property
    def unlabeled_epoch_index(self):
        if self.using_unlabeled_data:
            return self.global_step_index / self.num_unlabeled_steps_per_epoch
        return 0

    @property
    def batch_index(self):
        # type: () -> int

        """
        The index of the batch within the current epoch.
        Only works when using non-multiprocessing context.

        # Arguments
            None
        # Returns
            :return: index of the batch within the current epoch
        """

        return self.global_step_index % self.num_steps_per_epoch

    @property
    def unlabeled_batch_index(self):
        if self.using_unlabeled_data:
            return self.global_step_index % self.num_unlabeled_steps_per_epoch
        return 0

    @property
    def num_steps_per_epoch(self):
        # type: () -> int

        """
        Number of steps per epoch.

        # Arguments
            None
        # Returns
            :return: number of steps per epoch
        """

        return int(np.ceil(self.n_labeled / float(self.labeled_batch_size)))

    @property
    def num_unlabeled_steps_per_epoch(self):
        if self.using_unlabeled_data:
            return int(np.ceil(self.n_unlabeled / float(self.unlabeled_batch_size)))
        return 0

    @property
    def lock(self):
        # type: () -> multiprocessing.Lock

        """
        Multiprocessing lock. Applies to multi-processing as well as threading.

        # Arguments
            None
        # Returns
            :return: a multiprocessing/threading lock
        """

        if self._lock is None:
            self._lock = multiprocessing.Lock()

        return self._lock

    @property
    def using_unlabeled_data(self):
        # type: () -> bool

        """
        Boolean describing whether the iterator uses unlabeled data.

        # Arguments
            None
        # Returns
            :return: true if using unlabeled false otherwise.
        """

        return self.n_unlabeled > 0 and self.unlabeled_batch_size > 0

    @property
    def total_batch_size(self):
        # type: () -> int
        """
        How many samples are within a single batch in total (labeled + unlabeled)

        # Arguments
            None
        # Returns
            :return: number of samples per batch (may be smaller if data is uneven)
        """
        return self.labeled_batch_size + self.unlabeled_batch_size

    def __len__(self):
        """
        Length is defined as the length of the sequence which corresponds to the number of
        batches (steps) in a single epoch.

        # Arguments
            None
        # Returns
            :return: The number of batches (steps) per epoch
        """

        return self.num_steps_per_epoch

    @abstractmethod
    def get_batch(self, e_idx, b_idx):
        # type: (int, int) -> (list, list)
        """
        This method returns a batch when given the epoch index and batch index (within the epoch).
        Note: this method is deterministic i.e. always returns the same batch given the epoch index
        and the batch index.

        # Arguments
            :param e_idx: Epoch index
            :param b_idx: Batch index

        # Returns
            :return: A batch of data
        """

        raise NotImplementedError('This method is not implemented in the abstract DataSetIterator')

    @abstractmethod
    def next(self):
        # type: () -> (list, list)

        """
        This method returns the next batch in the sequence.

        # Arguments
            None
        # Returns
            :return: The next batch of data
        """

        raise NotImplementedError('This method is not implemented in the abstract DataSetIterator')

    def on_epoch_end(self):
        # type: () -> ()

        """
        A callback method that is called at the end of every epoch. This way we can e.g. shuffle
        the data between epochs.

        # Arguments
            None
        # Returns
            Nothing
        """
        raise NotImplementedError('This method is not implemented in the abstract DataSetIterator')

    def __next__(self, *args, **kwargs):
        # type: () -> (list, list)

        """
        This method returns the next batch in the sequence.

        # Arguments
            None
        # Returns
            :return: The next batch of data
        """

        return self.next(*args, **kwargs)

    def __iter__(self):
        # type: () -> (list, list)

        """
        This method is needed if we want to do something like: for x, y in data_gen.flow()

        # Arguments
            None
        # Returns
            :return: A batch of data as a tuple (X, Y) or (X, Y, S_WEIGHTS)
        """

        # Needed if we want to do something like:
        # for x, y in data_gen.flow()
        return self


class BasicDataSetIterator(DataSetIterator):
    """
    A class for iterating a basic data set with normal indexing and no special
    class balancing.
    """

    def __init__(self,
                 data_generator,
                 n_labeled,
                 n_unlabeled,
                 labeled_batch_size,
                 unlabeled_batch_size,
                 shuffle,
                 seed,
                 logger=None,
                 initial_epoch=0):
        # type: (DataGenerator, int, int, int, int, bool, int, Logger, int) -> None

        """
        # Arguments
            :param data_generator: DataGenerator, a data generator that returns batches of data when provided with batch indices
            :param n_labeled: integer, total number of labeled samples in the data set to loop over
            :param labeled_batch_size: integer, size of labeled data in batch
            :param n_unlabeled: integer, total number of unlabeled samples in the data set to loop over
            :param unlabeled_batch_size: integer, size of unlabeled data in batch
            :param shuffle: boolean, whether to shuffle the data between epochs.
            :param seed: random seeding for data shuffling.
            :param logger: logger instance for logging
            :param initial_epoch: initial epoch
        # Returns
            Nothing
        """

        super(BasicDataSetIterator, self).__init__(data_generator=data_generator,
                                                   n_labeled=n_labeled,
                                                   n_unlabeled=n_unlabeled,
                                                   labeled_batch_size=labeled_batch_size,
                                                   unlabeled_batch_size=unlabeled_batch_size,
                                                   shuffle=shuffle,
                                                   seed=seed,
                                                   logger=logger,
                                                   initial_epoch=initial_epoch)

        self.__index_generator = None

        # Create batch index buffers for N epochs
        self._labeled_batch_index_buffer = BatchIndexBuffer(n=n_labeled,
                                                            batch_size=labeled_batch_size,
                                                            shuffle=shuffle,
                                                            seed=seed,
                                                            logger=logger,
                                                            initial_epoch=self.initial_epoch)

        if self.using_unlabeled_data:
            self._unlabeled_batch_index_buffer = BatchIndexBuffer(n=n_unlabeled,
                                                                  batch_size=unlabeled_batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  logger=logger,
                                                                  initial_epoch=self.unlabeled_epoch_index)
        else:
            self._unlabeled_batch_index_buffer = None

    @property
    def _index_generator(self):
        if self.__index_generator is None:
            self.__index_generator = self._flow_index()

        return self.__index_generator

    def _flow_index(self):
        # type: () -> (list[int])

        """
        Generates batch indices continuously. Increases the global step index on every call.

        # Arguments
            None
        # Returns
            :return: A list of indices (for a data batch)
        """

        # Ensure we start from a clean table with batch idx at zero and global step idx
        # at 0
        self.reset()

        while 1:
            # Get labeled data
            labeled_batch = self._labeled_batch_index_buffer.get_batch_indices(e_idx=self.epoch_index, b_idx=self.batch_index)

            # Get unlabeled data if using unlabeled data
            if self.using_unlabeled_data:
                unlabeled_batch = self._unlabeled_batch_index_buffer.get_batch_indices(e_idx=self.unlabeled_epoch_index, b_idx=self.unlabeled_batch_index)
            else:
                unlabeled_batch = None

            self.global_step_index += 1

            yield labeled_batch, unlabeled_batch

    def get_batch(self, e_idx, b_idx):
        # type: (int, int) -> (list, list)

        if b_idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, but the Sequence has length {length}'.format(idx=b_idx, length=len(self)))

        # Calculate the global step index
        g_idx = self.num_steps_per_epoch * e_idx + b_idx

        # Get labeled data
        labeled_batch = self._labeled_batch_index_buffer.get_batch_indices(e_idx=e_idx, b_idx=b_idx)

        # Get unlabeled data
        if self.using_unlabeled_data:
            ul_steps_per_epoch = self.num_unlabeled_steps_per_epoch
            ul_e_idx = g_idx / ul_steps_per_epoch
            ul_b_idx = g_idx % ul_steps_per_epoch
            unlabeled_batch = self._unlabeled_batch_index_buffer.get_batch_indices(e_idx=ul_e_idx, b_idx=ul_b_idx)
        else:
            unlabeled_batch = None

        # Use the data generator to generate the data
        self.logger.debug_log('e_idx: {}, b_idx: {}, g_idx: {}, pid: {}, labeled: {}, ul: {}'.format(e_idx, b_idx, g_idx, os.getpid(), labeled_batch, unlabeled_batch))
        return self.data_generator.get_data_batch(step_idx=g_idx,
                                                  labeled_batch=labeled_batch,
                                                  unlabeled_batch=unlabeled_batch)

    def next(self):
        # type: () -> (list, list)

        """
        Legacy support for threaded iteration.

        # Arguments
            None
        # Returns
            :return: A batch of data
        """

        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            step_idx = self.global_step_index
            labeled_batch, unlabeled_batch = next(self._index_generator)

        # The data generation is not under lock so it can be carried out in parallel
        return self.data_generator.get_data_batch(step_idx=step_idx,
                                                  labeled_batch=labeled_batch,
                                                  unlabeled_batch=unlabeled_batch)

    def on_epoch_end(self):
        pass


class MaterialSampleIterationMode(Enum):
    UNIFORM_MAX = 0     # Sample each material class uniformly. Set the number of steps per epoch according to max class num samples.
    UNIFORM_MIN = 1     # Sample each material class uniformly. Set the number of steps per epoch according to min class num samples.
    UNIFORM_MEAN = 2    # Sample each class uniformly. Set the number of steps per epoch according to mean samples per class.
    UNIQUE = 3          # Iterate through all the unique samples once within epoch - means no balancing


class MaterialSampleDataSetIterator(DataSetIterator):
    """
    A class for iterating through MaterialSample data set as the labeled data set.
    """

    _NUM_QUEUED_EPOCHS = 4
    _SHARED_MATERIAL_CATEGORY_PIXELS_SEEN = {}
    _SHARED_MATERIAL_CATEGORY_NEXT_SAMPLE_INDICES = {}

    def __init__(self,
                 data_generator,
                 material_samples,
                 n_unlabeled,
                 labeled_batch_size,
                 unlabeled_batch_size,
                 shuffle,
                 seed,
                 logger=None,
                 initial_epoch=0,
                 iteration_mode=MaterialSampleIterationMode.UNIFORM_MEAN,
                 balance_pixel_samples=False):

        super(MaterialSampleDataSetIterator, self).__init__(
            data_generator=data_generator,
            n_labeled=MaterialSampleDataSetIterator._get_n_labeled(material_samples=material_samples, iteration_mode=iteration_mode),
            n_unlabeled=n_unlabeled,
            labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            shuffle=shuffle,
            seed=seed,
            logger=logger,
            initial_epoch=initial_epoch)

        self._iteration_mode = iteration_mode
        self._balance_pixel_samples = balance_pixel_samples
        self.__index_generator = None

        # Sanity check: per-pixel sample balancing only makes sense if not using unique iteration mode
        if self._iteration_mode == MaterialSampleIterationMode.UNIQUE and self._balance_pixel_samples:
            raise ValueError('Cannot use per pixel balancing with MaterialSampleIterationMode UNIQUE')

        """
        Labeled data (MaterialSample data) iteration initialisation
        """

        # Create index arrays for material samples
        self._material_samples = []             # 2D array which holds all the material samples, each category in their own index
        self._material_samples_flattened = []   # 1D array which holds all the material samples as tuples (category_idx, sample_idx)

        for i, material_category in enumerate(material_samples):
            material_category_index = i
            n_samples_in_material_category = len(material_category)

            # 2D material samples array
            if self.shuffle:
                self._material_samples.append(np.random.permutation(n_samples_in_material_category))
            else:
                self._material_samples.append(np.arange(n_samples_in_material_category))

            # 1D flattened material samples array
            for sample_index in range(0, n_samples_in_material_category):
                self._material_samples_flattened.append((material_category_index, sample_index))

        if self.shuffle:
            random.shuffle(self._material_samples_flattened)

        # Create sampling probability map - describes the probability of sampling each category when using
        # iteration mode UNIFORM_MAX, UNIFORM_MIN or UNIFORM_MEAN
        self._num_material_samples = len(self._material_samples_flattened)
        self._num_material_categories = len(material_samples)
        self._num_non_zero_material_categories = sum(1 for material_category in material_samples if len(material_category) > 0)
        self._material_category_sampling_probabilities = np.zeros(self._num_material_categories, dtype=np.float64)

        # Calculate the sampling probabilities for each material category - for uniform sampling all non-zero material
        # categories should have the same probabilities
        material_category_sampling_probability = 1.0 / self._num_non_zero_material_categories

        for i in range(len(material_samples)):
            num_samples_in_category = len(material_samples[i])
            self._material_category_sampling_probabilities[i] = material_category_sampling_probability if num_samples_in_category > 0 else 0.0

            # Zero is assumed as the background class and should/can have zero instances
            if num_samples_in_category == 0 and i != 0:
                self.logger.warn('Material class {} has 0 material samples'.format(i))

        self.logger.debug_log('Material category sampling probabilities: {}'.format(self._material_category_sampling_probabilities))

        # Keep track of the next material sample index for each material category: 1D array with ints
        self._material_category_ignore_mask = np.equal(self._material_category_sampling_probabilities, 0.0)
        self._sampling_probability_update_mask = np.logical_not(self._material_category_ignore_mask)

        if self._material_category_next_sample_indices is None:
            self._material_category_next_sample_indices = Array('i', self._num_material_categories)

        if self._balance_pixel_samples and self._material_category_pixels_seen is None:
            self._material_category_pixels_seen = Array(ctypes.c_uint64, self._num_material_categories)

        # Create an epoch queue for the material samples
        self._material_sample_epoch_queue = ExtendedDictionary(default=None)

        # We only need to generate data if we shuffle between epochs
        # otherwise the data is just the default return value for the epoch queue
        if not self._balance_pixel_samples:
            for i in range(0, MaterialSampleDataSetIterator._NUM_QUEUED_EPOCHS):
                e_idx = self.initial_epoch+i
                self._material_sample_epoch_queue[e_idx] = self._create_material_sample_index_array_for_epoch(e_idx=e_idx)

        """
        Unlabeled data iteration initialization
        """

        # Create index buffer for unlabeled data indices
        if self.using_unlabeled_data:
            self._unlabeled_batch_index_buffer = BatchIndexBuffer(n=n_unlabeled,
                                                                  batch_size=unlabeled_batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  logger=logger,
                                                                  initial_epoch=self.unlabeled_epoch_index,
                                                                  num_queued_epochs=MaterialSampleDataSetIterator._NUM_QUEUED_EPOCHS)
        else:
            self._unlabeled_batch_index_buffer = None

    @property
    def num_material_samples(self):
        # type: () -> int

        """
        Returns the total number of unique material samples

        # Returns
            :return: the number of material samples
        """

        return self._num_material_samples

    @property
    def _material_category_pixels_seen(self):
        # type: () -> Array
        return MaterialSampleDataSetIterator._SHARED_MATERIAL_CATEGORY_PIXELS_SEEN.get(self.uuid)

    @_material_category_pixels_seen.setter
    def _material_category_pixels_seen(self, value):
        # type: (Array) -> None
        MaterialSampleDataSetIterator._SHARED_MATERIAL_CATEGORY_PIXELS_SEEN[self.uuid] = value

    @property
    def _material_category_next_sample_indices(self):
        # type: () -> Array
        return MaterialSampleDataSetIterator._SHARED_MATERIAL_CATEGORY_NEXT_SAMPLE_INDICES.get(self.uuid)

    @_material_category_next_sample_indices.setter
    def _material_category_next_sample_indices(self, value):
        # type: (Array) -> None
        MaterialSampleDataSetIterator._SHARED_MATERIAL_CATEGORY_NEXT_SAMPLE_INDICES[self.uuid] = value

    def _calculate_pixels_per_material_in_batch(self, batch_data):

        """
        Batch data is a tuple of (inputs, outputs), which are both lists. This is the data
        directly from the generator and each array may contain multiple entries.

        # Arguments
            :param batch_data: the batch data (inputs, outputs)
        # Returns
            :return: the number of pixels in each material category in the batch
        """

        batch_inputs = batch_data[0]
        batch_outputs = batch_data[1]
        
        if self.data_generator.batch_data_format == BatchDataFormat.SUPERVISED:
            y = batch_outputs[0]
        elif self.data_generator.batch_data_format == BatchDataFormat.SEMI_SUPERVISED:
            # In the semi-supervised scenario we have to also discard the unlabeled data
            # from the calculation - assume its the N last elements of ground truth masks
            y = batch_inputs[1][0:self.labeled_batch_size]
        else:
            raise ValueError('Unknown batch data format: {}'.format(self.data_generator.batch_data_format))

        # Calculate the number of pixels in the different material categories
        # Note: do not modify y only get views into y
        y_flattened = y.squeeze().reshape(-1)
        pixels_per_material = np.bincount(y_flattened, minlength=self._num_material_categories).astype(dtype=np.uint64)
        return pixels_per_material

    @staticmethod
    def _get_n_labeled(material_samples, iteration_mode):
        # type: (list, MaterialSampleIterationMode) -> int

        """
        Returns the correct number of labeled samples according to material sample array and iteration mode.

        # Arguments
            :param material_samples: 2D list of material samples i.e. N_CATEGORIES with N_SAMPLES
            :param iteration_mode: the iteration mode for the samples
        # Returns
            :return: number of labeled samples
        """
        num_non_zero_material_classes = sum(1 for material_category in material_samples if len(material_category) > 0)

        # Iterate through all the unique material samples on every epoch
        if iteration_mode == MaterialSampleIterationMode.UNIQUE:
            return sum([len(material_category) for material_category in material_samples])
        # Iterate according to the maximum material class size
        elif iteration_mode == MaterialSampleIterationMode.UNIFORM_MAX:
            samples_per_material_category = max([len(material_category) for material_category in material_samples if len(material_category) > 0])
        # Iterate according to the minimum material class size
        elif iteration_mode == MaterialSampleIterationMode.UNIFORM_MIN:
            samples_per_material_category = min([len(material_category) for material_category in material_samples if len(material_category) > 0])
        # Iterate according to the mean material class size
        elif iteration_mode == MaterialSampleIterationMode.UNIFORM_MEAN:
            samples_per_material_category = int(np.mean(np.array([len(material_category) for material_category in material_samples if len(material_category) > 0])))
        else:
            raise ValueError('Unknown iteration mode: {}'.format(iteration_mode))

        return samples_per_material_category * num_non_zero_material_classes

    def _update_material_category_sampling_probabilities(self, pixels_seen_per_category):
        # type: (np.ndarray) -> None

        """
        Updates the material category sampling probabilities according to new pixels seen per category.
        Accumulates the parameter value to existing pixels seen value.

        # Arguments
            :param pixels_seen_per_category: new pixels seen per category expecting an iterable with N_CLASSES entries
        # Returns
            Nothing
        """

        if self._balance_pixel_samples:

            if len(pixels_seen_per_category) != self._num_material_categories:
                raise ValueError('Pixels seen per category dimension does not match with num categories: {} vs {}'
                                 .format(len(pixels_seen_per_category), self._num_material_categories))

            # Accumulate to the shared stored value
            pixels_seen_per_category = pixels_seen_per_category.astype(np.uint64)
            total_pixels_seen_per_category = np.zeros_like(pixels_seen_per_category, dtype=np.uint64)

            for i in range(0, len(pixels_seen_per_category)):
                if self._material_category_ignore_mask[i]:
                    self._material_category_pixels_seen[i] = 0
                else:
                    self._material_category_pixels_seen[i] += int(pixels_seen_per_category[i])
                    total_pixels_seen_per_category[i] = self._material_category_pixels_seen[i]

            # Make sure no category has zeros for the coming division
            total_pixels_seen_per_category = np.clip(total_pixels_seen_per_category, 1, np.iinfo(total_pixels_seen_per_category.dtype).max)

            # Calculate the inverse proportions
            inv_proportions = np.sum(total_pixels_seen_per_category).astype(np.float64)/total_pixels_seen_per_category.astype(np.float64)
            inv_proportions[self._material_category_ignore_mask] = 0.0

            # Scale so that everything sums to one and ensure that the ignored classes have a probability of 0
            updated_probabilities = inv_proportions/np.sum(inv_proportions)
            self._material_category_sampling_probabilities = updated_probabilities
            self._material_category_sampling_probabilities[self._material_category_ignore_mask] = 0.0

            # Sanity check
            self.logger.debug_log('Material category sampling probabilities updated to: {} - sum: {}'
                                  .format(self._material_category_sampling_probabilities, np.sum(self._material_category_sampling_probabilities)))

        else:
            self.logger.warn('Updating material category sampling probabilities when balance_pixel_samples is False - using uniform class probabilities')

            material_category_sampling_probability = 1.0 / self._num_non_zero_material_categories

            for i in range(len(self._material_samples)):
                num_samples_in_category = len(self._material_samples[i])
                self._material_category_sampling_probabilities[i] = material_category_sampling_probability if num_samples_in_category > 0 else 0.0

    def _get_next_sample_for_material_category(self, material_category_idx):
        # type: (int) -> int

        """
        Returns the next material sample index for the given material category.

        # Arguments
            :param material_category_idx: category index [0, N_CATEGORIES]
        # Returns
            :return: the next sample index for the given material category
        """

        # Get the current sample index for the material category
        material_sample_idx = self._material_category_next_sample_indices[material_category_idx]

        # Update the next sample index for the material category
        next_material_sample_idx_for_category = material_sample_idx + 1

        # If all of the samples in the category have been used, zero out the
        # index for the category and shuffle the category list if shuffle is enabled
        if next_material_sample_idx_for_category >= len(self._material_samples[material_category_idx]):
            self._material_category_next_sample_indices[material_category_idx] = 0

            if self.shuffle:
                self._material_samples[material_category_idx] = np.random.permutation(len(self._material_samples[material_category_idx]))
            else:
                self._material_samples[material_category_idx] = np.arange(len(self._material_samples[material_category_idx]))
        else:
            self._material_category_next_sample_indices[material_category_idx] = next_material_sample_idx_for_category

        return material_sample_idx

    def _get_material_sample_index_array_for_epoch(self, e_idx):
        # type: (int) -> np.ndarray

        """
        Generates the batch material sample indices for a specific epoch.

        # Arguments
            :param e_idx: epoch index
        # Returns
            :return: sample indices for the specified epoch
        """

        # If we are not shuffling the epoch queue is always the same which
        # should be the default value
        if e_idx not in self._material_sample_epoch_queue and self.shuffle:
            self._update_material_sample_epoch_queue(r_e_idx=e_idx)

        return self._material_sample_epoch_queue[e_idx]

    def _create_material_sample_index_array_for_epoch(self, e_idx):
        # type: (int) -> list[(int, int)]

        """
        Generates the batch material sample indices for a specific epoch.

        # Arguments
            :param e_idx: epoch index
        # Returns
            :return: material sample indices for the specified epoch
        """

        # Use the same random seed specified by the epoch as the base
        # to guarantee deterministic runs
        np.random.seed(self.seed + e_idx)

        if self._balance_pixel_samples:
            raise ValueError('Cannot pre-generate pixel balanced index arrays for epoch')

        # If we are iterating through all the unique material samples on each epoch
        if self._iteration_mode == MaterialSampleIterationMode.UNIQUE:
            if self.shuffle:
                index_array = np.random.permutation(self.n_labeled)
            else:
                index_array = np.arange(self.n_labeled)

            # Use the index array to index the flattened material sample array
            material_sample_index_array = [self._material_samples_flattened[i] for i in index_array]
        # If we are iterating through material samples in a category balanced way
        elif self._iteration_mode == MaterialSampleIterationMode.UNIFORM_MEAN or \
             self._iteration_mode == MaterialSampleIterationMode.UNIFORM_MAX or \
             self._iteration_mode == MaterialSampleIterationMode.UNIFORM_MIN:

            # Get categories for each sample according to the sampling probabilities
            material_sample_categories = np.random.choice(a=self._num_material_categories, size=self.n_labeled, p=self._material_category_sampling_probabilities)
            material_sample_index_array = []

            # Get the material samples for the categories
            for material_category_idx in material_sample_categories:
                material_sample_idx = self._get_next_sample_for_material_category(material_category_idx=material_category_idx)
                material_sample_index_array.append((material_category_idx, material_sample_idx))
        else:
            raise ValueError('Unknown iteration mode: {}'.format(self._iteration_mode))

        return material_sample_index_array

    def _update_material_sample_epoch_queue(self, r_e_idx):
        # type: (int, int) -> ()

        """
        Moves the epoch queue window ahead by half the number of queued epochs.
        For example if the queue had epochs 0,1,2,3 after this call the queue
        would have epochs 2,3,4,5.

        # Arguments
            :param r_e_idx: requested epoch index
        # Returns
            Nothing
        """

        keys = self._material_sample_epoch_queue.keys()
        keys.sort()
        max_key = max(keys)
        min_key = min(keys)
        window_move_size = len(keys) / 2

        # Check whether there is a bug and we are trying to request an epoch from the past - warn but don't crash
        if r_e_idx < min_key:
            self.logger.warn('Requested a past epoch: min key: {}, requested epoch: {}'.format(min_key, r_e_idx))
            self._material_sample_epoch_queue[r_e_idx] = self._create_material_sample_index_array_for_epoch(e_idx=r_e_idx)
        # Otherwise move the queued epochs window by half forward
        else:
            # Remove the old first half of the queued epochs
            for i in range(0, window_move_size):
                k = keys[i]
                if k in self._material_sample_epoch_queue:
                    del self._material_sample_epoch_queue[k]

            # Add a new half in to the queued epochs
            for i in range(0, window_move_size):
                k = max_key+i+1
                self._material_sample_epoch_queue[k] = self._create_material_sample_index_array_for_epoch(e_idx=k)

        # Final sanity check
        if r_e_idx not in self._material_sample_epoch_queue:
            self.logger.warn('Requested for an epoch ({}) outside the sliding window distance: {}'.format(r_e_idx, window_move_size))
            self._material_sample_epoch_queue[r_e_idx] = self._create_material_sample_index_array_for_epoch(e_idx=r_e_idx)

    def _get_material_sample_batch_indices(self, e_idx, b_idx):
        # type: (int, int) -> list[(int, int)]

        """
        Returns the material sample indices for a given epoch and batch index as a numpy array.

        # Arguments
            :param e_idx: epoch index
            :param b_idx: batch index (within the epoch)
        # Returns
            :return: material sample indices for the given epoch
        """

        # If we use per pixel balancing we can't pre calculate samples for epoch
        # then samples are created on per batch basis
        if self._balance_pixel_samples:
            g_idx = e_idx * self.num_steps_per_epoch + b_idx
            np.random.seed(self.seed + g_idx)

            # Get categories for each sample according to the sampling probabilities
            material_sample_categories = np.random.choice(a=self._num_material_categories, size=self.labeled_batch_size, p=self._material_category_sampling_probabilities)
            material_sample_index_array = []

            # Get the material samples for the categories
            for material_category_idx in material_sample_categories:
                material_sample_idx = self._get_next_sample_for_material_category(material_category_idx=material_category_idx)
                material_sample_index_array.append((material_category_idx, material_sample_idx))

            batch = material_sample_index_array
        else:
            # Batch indices
            material_sample_index_array = self._get_material_sample_index_array_for_epoch(e_idx=e_idx)
            current_sample_index = (b_idx * self.labeled_batch_size) % self.n_labeled

            if self.n_labeled > b_idx + self.labeled_batch_size:
                samples_in_batch = self.labeled_batch_size
                is_last_batch = False
            else:
                samples_in_batch = self.n_labeled - current_sample_index
                is_last_batch = True

            batch = material_sample_index_array[current_sample_index:current_sample_index + samples_in_batch]

        return batch

    @property
    def _index_generator(self):
        if self.__index_generator is None:
            self.__index_generator = self._flow_index()

        return self.__index_generator

    def _flow_index(self):
        # type: () -> (list[int])

        """
        Generates batch indices continuously. Increases the global step index on every call.

        # Arguments
            None
        # Returns
            :return: A list of indices (for a data batch)
        """

        # Ensure we start from a clean table with batch idx at zero and global step idx
        # at 0
        self.reset()

        while 1:
            labeled_batch = self._get_material_sample_batch_indices(e_idx=self.epoch_index, b_idx=self.batch_index)

            # Get unlabeled data if using unlabeled data
            if self.using_unlabeled_data:
                unlabeled_batch = self._unlabeled_batch_index_buffer.get_batch_indices(e_idx=self.unlabeled_epoch_index, b_idx=self.unlabeled_batch_index)
            else:
                unlabeled_batch = None

            self.global_step_index += 1

            yield labeled_batch, unlabeled_batch

    def get_batch(self, e_idx, b_idx):
        # type: (int, int) -> (list, list)

        """
        Returns a batch of data given an epoch index and a batch index (within the epoch).

        # Arguments
            :param e_idx: Epoch index
            :param b_idx: Batch index within the epoch
        # Returns
            :return: A batch of data
        """

        if b_idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, but the Sequence has length {length}'.format(idx=b_idx, length=len(self)))

        # Calculate the global step index
        g_idx = (self.num_steps_per_epoch * e_idx) + b_idx

        # Get labeled data
        labeled_batch = self._get_material_sample_batch_indices(e_idx=e_idx, b_idx=b_idx)

        # Get unlabeled data
        if self.using_unlabeled_data:
            ul_steps_per_epoch = self.num_unlabeled_steps_per_epoch
            ul_e_idx = g_idx / ul_steps_per_epoch
            ul_b_idx = g_idx % ul_steps_per_epoch
            unlabeled_batch = self._unlabeled_batch_index_buffer.get_batch_indices(e_idx=ul_e_idx, b_idx=ul_b_idx)
        else:
            unlabeled_batch = None

        # Use the data generator to generate the data
        self.logger.debug_log('e_idx: {}, b_idx: {}, g_idx: {}, pid: {}, labeled: {}, ul: {}'.format(e_idx, b_idx, g_idx, os.getpid(), labeled_batch, unlabeled_batch))

        batch_data = self.data_generator.get_data_batch(step_idx=g_idx,
                                                        labeled_batch=labeled_batch,
                                                        unlabeled_batch=unlabeled_batch)

        if self._balance_pixel_samples:
            pixels_per_material_category = self._calculate_pixels_per_material_in_batch(batch_data=batch_data)
            self.logger.debug_log('e_idx: {}, b_idx: {}, material pixels: {}'.format(e_idx, b_idx, list(pixels_per_material_category)))
            self._update_material_category_sampling_probabilities(pixels_per_material_category)

        return batch_data

    def next(self):
        # type: () -> (list, list)

        """
        Legacy support for threaded iteration.

        # Arguments
            None
        # Returns
            :return: A batch of data
        """

        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            step_idx = self.global_step_index
            labeled_batch, unlabeled_batch = next(self._index_generator)

        # The data generation is not under lock so it can be carried out in parallel
        batch_data = self.data_generator.get_data_batch(step_idx=step_idx,
                                                        labeled_batch=labeled_batch,
                                                        unlabeled_batch=unlabeled_batch)

        if self._balance_pixel_samples:
            with self.lock:
                pixels_per_material_category = self._calculate_pixels_per_material_in_batch(batch_data=batch_data)
                self.logger.debug_log('material pixels: {}'.format(list(pixels_per_material_category)))
                self._update_material_category_sampling_probabilities(pixels_per_material_category)
                self.logger.debug_log('updated material pixels: {}'.format(list(self._material_category_pixels_seen)))

        return batch_data

    def on_epoch_end(self):
        pass
