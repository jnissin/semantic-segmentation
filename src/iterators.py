# coding=utf-8

import numpy as np
import threading
import os
import random

from keras_extensions.utils.data_utils import Sequence

from abc import ABCMeta, abstractmethod
from enum import Enum

from logger import Logger
from utils.dataset_utils import MaterialSample


class ExtendedDictionary(dict):
    def __init__(self, default, **kwargs):
        self.default = default
        super(ExtendedDictionary, self).__init__(**kwargs)

    def __getitem__(self, key):
        if key in self:
            return super(ExtendedDictionary, self).__getitem__(key)
        return self.default


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

        # The following member variables only work when not used in multiprocessing context
        self.global_step_index = self.initial_epoch * self.num_steps_per_epoch  # The global step index (how many batches have been processed altogether)
        self._lock = None

        # Handle the initial random seeding
        np.random.seed(self.seed)
        random.seed(self.seed)

    def reset(self):
        self.global_step_index = self.initial_epoch * self.num_steps_per_epoch

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
            return self.global_step_index / self.unlabeled_steps_per_epoch
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
            return self.global_step_index % self.unlabeled_steps_per_epoch
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
    def unlabeled_steps_per_epoch(self):
        if self.using_unlabeled_data:
            return int(np.ceil(self.n_unlabeled / float(self.unlabeled_batch_size)))
        return 0

    @property
    def lock(self):
        # type: () -> threading.Lock

        """
        Threading lock. Only works when using non-multiprocessing context. Otherwise,
        should not be accessed.

        # Arguments
            None
        # Returns
            :return: a thread lock
        """

        if self._lock is None:
            self._lock = threading.Lock()

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

    QUEUED_EPOCHS = 4

    def __init__(self, data_generator, n_labeled, n_unlabeled, labeled_batch_size, unlabeled_batch_size, shuffle, seed, logger=None, initial_epoch=0):
        # type: (DataGenerator, int, int, int, int, bool, int, Logger, int) -> None

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

        # Generate the initial data
        self.labeled_epoch_queue = ExtendedDictionary(default=np.arange(n_labeled) if not self.shuffle else None)
        self.unlabeled_epoch_queue = ExtendedDictionary(default=np.arange(n_unlabeled) if not self.shuffle else None)

        # We only need to generate data if we shuffle between epochs
        if self.shuffle:
            for i in range(0, BasicDataSetIterator.QUEUED_EPOCHS):
                e_idx = self.initial_epoch+i
                self.labeled_epoch_queue[e_idx] = self._create_index_array_for_epoch(e_idx=e_idx, n=self.n_labeled)

                # Note: unlabeled data follows separate indexing
                if self.using_unlabeled_data:
                    ul_e_idx = self.unlabeled_epoch_index + i
                    self.unlabeled_epoch_queue[ul_e_idx] = self._create_index_array_for_epoch(e_idx=ul_e_idx, n=self.n_unlabeled)

    @property
    def _index_generator(self):
        if self.__index_generator is None:
            self.__index_generator = self._flow_index()

        return self.__index_generator

    def _get_index_array_for_epoch(self, epoch_queue, e_idx, n):
        # type: (ExtendedDictionary, int, int, bool) -> np.ndarray

        """
        Generates the batch sample indices for a specific epoch.

        # Arguments
            :param epoch_queue: a dictionary of epoch indices to sample index arrays
            :param e_idx: epoch index
            :param n: number of samples in batch
        # Returns
            :return: sample indices for the specified epoch
        """

        if e_idx not in epoch_queue and self.shuffle:
            self._update_epoch_queue(epoch_queue=epoch_queue, n=n, r_e_idx=e_idx)

        return epoch_queue[e_idx]

    def _create_index_array_for_epoch(self, e_idx, n):
        # type: (int, int, bool) -> np.ndarray

        """
        Generates the batch sample indices for a specific epoch.

        # Arguments
            :param e_idx: epoch index
            :param n: number of samples in batch
        # Returns
            :return: sample indices for the specified epoch
        """

        if self.shuffle:
            np.random.seed(self.seed + e_idx)
            index_array = np.random.permutation(n)
        else:
            index_array = np.arange(n)

        return index_array

    def _update_epoch_queue(self, epoch_queue, n, r_e_idx):
        # type: (ExtendedDictionary, int, int) -> ()

        """
        Moves the epoch queue window ahead by half the number of queued epochs.
        For example if the queue had epochs 0,1,2,3 after this call the queue
        would have epochs 2,3,4,5.

        # Arguments
            :param epoch_queue: the epoch queue
            :param n: number of elements per epoch
            :param r_e_idx: requested epoch index
        # Returns
            Nothing
        """
        keys = epoch_queue.keys()
        keys.sort()
        max_key = max(keys)
        min_key = min(keys)
        window_move_size = len(keys) / 2

        # Check whether there is a bug and we are trying to request an epoch from the past - warn but don't crash
        if r_e_idx < min_key:
            self.logger.warn('Requested a past epoch: min key: {}, requested epoch: {}'.format(min_key, r_e_idx))
            epoch_queue[r_e_idx] = self._create_index_array_for_epoch(e_idx=r_e_idx, n=n)
        # Otherwise move the queued epochs window by half forward
        else:
            # Remove the old first half of the queued epochs
            for i in range(0, window_move_size):
                k = keys[i]
                if k in epoch_queue:
                    del epoch_queue[k]

            # Add a new half in to the queued epochs
            for i in range(0, window_move_size):
                k = max_key+i+1
                epoch_queue[k] = self._create_index_array_for_epoch(e_idx=k, n=n)

        # Final sanity check
        if r_e_idx not in epoch_queue:
            self.logger.warn('Requested for an epoch ({}) outside the sliding window distance: {}'.format(r_e_idx, window_move_size))
            epoch_queue[r_e_idx] = self._create_index_array_for_epoch(e_idx=r_e_idx, n=n)

    def _get_batch_parameters(self, n, batch_size, idx):
        current_index = (idx * batch_size) % n
        is_last_batch = False

        if n > idx + batch_size:
            samples_in_batch = batch_size
        else:
            samples_in_batch = n - current_index
            is_last_batch = True

        return current_index, samples_in_batch, is_last_batch

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
            b_idx = self.batch_index
            e_idx = self.epoch_index

            # Get labeled data
            labeled_index_array = self._get_index_array_for_epoch(self.labeled_epoch_queue, e_idx=e_idx, n=self.n_labeled)
            labeled_current_index, labeled_samples_in_batch, _ = self._get_batch_parameters(self.n_labeled, self.labeled_batch_size, b_idx)
            labeled_batch = labeled_index_array[labeled_current_index:labeled_current_index + labeled_samples_in_batch]

            # Get unlabeled data
            if self.using_unlabeled_data:
                unlabeled_index_array = self._get_index_array_for_epoch(epoch_queue=self.unlabeled_epoch_queue, e_idx=self.unlabeled_epoch_index, n=self.n_unlabeled)
                ul_current_index, ul_samples_in_batch, ul_batch = self._get_batch_parameters(self.n_unlabeled, self.unlabeled_batch_size, self.unlabeled_batch_index)
                unlabeled_batch = unlabeled_index_array[ul_current_index:ul_current_index + ul_samples_in_batch]
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
        labeled_index_array = self._get_index_array_for_epoch(self.labeled_epoch_queue, e_idx=e_idx, n=self.n_labeled)
        labeled_current_index, labeled_samples_in_batch, _ = self._get_batch_parameters(self.n_labeled, self.labeled_batch_size, b_idx)
        labeled_batch = labeled_index_array[labeled_current_index:labeled_current_index + labeled_samples_in_batch]

        # Get unlabeled data
        if self.using_unlabeled_data:
            ul_steps_per_epoch = int(np.ceil(self.n_unlabeled / float(self.unlabeled_batch_size)))
            ul_b_idx = g_idx % ul_steps_per_epoch
            ul_e_idx = g_idx/ul_steps_per_epoch
            unlabeled_index_array = self._get_index_array_for_epoch(epoch_queue=self.unlabeled_epoch_queue, e_idx=ul_e_idx, n=self.n_unlabeled)
            ul_current_index, ul_samples_in_batch, ul_batch = self._get_batch_parameters(self.n_unlabeled, self.unlabeled_batch_size, ul_b_idx)
            unlabeled_batch = unlabeled_index_array[ul_current_index:ul_current_index + ul_samples_in_batch]
        else:
            unlabeled_batch = None

        # Use the data generator to generate the data
        self.logger.debug_log('b_idx: {}, e_idx: {}, g_idx: {}, pid: {}'.format(b_idx, e_idx, g_idx, os.getpid()))
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
    A class for iterating randomly through MaterialSamples for a data set in batches.
    """

    def __init__(self, material_samples, batch_size, shuffle, seed, logger, iter_mode=MaterialSampleIterationMode.UNIFORM_MAX):
        # type: (list[list[MaterialSample]], int, bool, int, Logger, MaterialSampleIterationMode) -> None

        self._num_unique_material_samples = sum(len(material_category) for material_category in material_samples)
        super(MaterialSampleDataSetIterator, self).__init__(n=self._num_unique_material_samples, batch_size=batch_size, shuffle=shuffle, seed=seed, logger=logger)

        # Calculate uniform probabilities for all classes that have non zero samples
        self.iter_mode = iter_mode
        self._material_category_sampling_probabilities = np.zeros(len(material_samples), dtype=np.float64)
        self._num_non_zero_classes = sum(1 for material_category in material_samples if len(material_category) > 0)

        self.logger.debug_log('Samples per material category: {}'.format([len(material_category) for material_category in material_samples]))

        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            self._samples_per_material_category_per_epoch = None
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MAX:
            self._samples_per_material_category_per_epoch = max([len(material_category) for material_category in material_samples if len(material_category) > 0])
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MIN:
            self._samples_per_material_category_per_epoch = min([len(material_category) for material_category in material_samples if len(material_category) > 0])
        elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MEAN:
            self._samples_per_material_category_per_epoch = int(np.mean(np.array([len(material_category) for material_category in material_samples if len(material_category) > 0])))
        else:
            raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

        # Build index lists for the different material samples
        self._material_samples = []

        for material_category in material_samples:
            if not shuffle:
                self._material_samples.append(np.arange(len(material_category)))
            else:
                self._material_samples.append(np.random.permutation(len(material_category)))

        # Build a flattened list of all the material samples (for unique iteration)
        self._material_samples_flattened = []

        for i in range(len(self._material_samples)):
            for j in range(len(self._material_samples[i])):
                self._material_samples_flattened.append((i, j))

        # Calculate the sampling probabilities for each class - for uniform sampling all non-zero material
        # categories should have the same probabilities
        material_category_sampling_probability = 1.0 / self._num_non_zero_classes

        for i in range(len(material_samples)):
            num_samples_in_category = len(material_samples[i])

            if num_samples_in_category > 0:
                self._material_category_sampling_probabilities[i] = material_category_sampling_probability
            else:
                # Zero is assumed as the background class and should/can have zero instances
                if i != 0:
                    self.logger.warn('Material class {} has 0 material samples'.format(i))

                self._material_category_sampling_probabilities[i] = 0.0

        self.logger.debug_log('Material category sampling probabilities: {}'.format(self._material_category_sampling_probabilities))

        # Keep track of the current sample (next sample to be given) in each material category
        self.num_material_classes = len(material_samples)
        self._current_samples = [0] * self.num_material_classes
        self._material_category_pixels_seen = np.zeros(self.num_material_classes, dtype=np.uint64)
        self._material_category_ignore_mask = np.equal(self._material_category_sampling_probabilities, 0.0)
        self._sampling_probability_update_mask = np.logical_not(self._material_category_ignore_mask)

    def get_next_batch(self, idx=None):
        # type: () -> (list[tuple[int]], int, int)

        """
        Gives the next batch as tuple of indices to the material samples 2D array.
        The tuples are in the form (sample_category_idx, sample_idx).

        # Arguments
            Nothing
        # Returns
            :return: a batch of material samples as tuples of indices into 2D material sample array
        """

        with self.lock:
            super(MaterialSampleDataSetIterator, self).get_next_batch()

            if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
                return self._get_next_batch_unique()
            elif self.iter_mode == MaterialSampleIterationMode.UNIFORM_MAX or \
                 self.iter_mode == MaterialSampleIterationMode.UNIFORM_MIN or \
                 self.iter_mode == MaterialSampleIterationMode.UNIFORM_MEAN:
                return self._get_next_batch_uniform()

            raise ValueError('Unknown iteration mode: {}'.format(self.iter_mode))

    def update_sampling_probabilities(self, pixels_seen_per_category):
        # type: (np.ndarray) -> None

        with self.lock:
            if len(pixels_seen_per_category) != self.num_material_classes:
                raise ValueError('Pixels seen per category dimension does not match with num classes: {} vs {}'
                                 .format(len(pixels_seen_per_category), self.num_material_classes))

            # Accumulate to the stored value
            pixels_seen_per_category = pixels_seen_per_category.astype(np.uint64)
            pixels_seen_per_category[self._material_category_ignore_mask] = 0
            self._material_category_pixels_seen += pixels_seen_per_category
            self._material_category_pixels_seen[self._material_category_ignore_mask] = 0

            # Make sure no category has zeros for the coming division
            total_pixels_seen_per_category = self._material_category_pixels_seen + pixels_seen_per_category
            total_pixels_seen_per_category = np.clip(total_pixels_seen_per_category, 1, np.iinfo(total_pixels_seen_per_category.dtype).max)

            # Calculate the inverse proportions
            inv_proportions = np.sum(total_pixels_seen_per_category).astype(np.float64)/total_pixels_seen_per_category.astype(np.float64)
            inv_proportions[self._material_category_ignore_mask] = 0.0

            # Scale so that everything sums to one and ensure that the ignored classes have a probability of 0
            updated_probabilities = inv_proportions/np.sum(inv_proportions)
            self._material_category_sampling_probabilities = updated_probabilities
            self._material_category_sampling_probabilities[self._material_category_ignore_mask] = 0.0

            # Sanity check
            self.logger.debug_log('Material category pixels seen: {}'.format(self._material_category_pixels_seen))
            self.logger.debug_log('Material category sampling probabilities updated to: {} - sum {}'.format(self._material_category_sampling_probabilities, np.sum(self._material_category_sampling_probabilities)))

    def _get_next_batch_uniform(self):
        sample_categories = np.random.choice(a=self.num_material_classes, size=self.batch_size, p=self._material_category_sampling_probabilities)
        batch = []

        for sample_category_idx in sample_categories:
            internal_sample_idx = self._current_samples[sample_category_idx]
            sample_idx = self._material_samples[sample_category_idx][internal_sample_idx]
            batch.append((sample_category_idx, sample_idx))

            # Keep track of the used samples in each category
            self._current_samples[sample_category_idx] += 1

            # If all of the samples in the category have been used, zero out the
            # index for the category and shuffle the category list if shuffle is enabled
            if self._current_samples[sample_category_idx] >= len(self._material_samples[sample_category_idx]):
                self._current_samples[sample_category_idx] = 0

                if self.shuffle:
                    self._material_samples[sample_category_idx] = np.random.permutation(len(self._material_samples[sample_category_idx]))
                else:
                    self._material_samples[sample_category_idx] = np.arange(len(self._material_samples[sample_category_idx]))

        n_samples = self._samples_per_material_category_per_epoch * self._num_non_zero_classes
        current_index = (self.batch_index * self.batch_size) % n_samples
        current_batch_size = len(batch)

        if n_samples > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        self.global_step_index += 1

        return batch, current_index, current_batch_size, self.global_step_index

    def _get_next_batch_unique(self):
        if self.batch_index == 0 and self.shuffle:
            np.random.shuffle(self._material_samples_flattened)

        current_index = (self.batch_index * self.batch_size) % self.n_labeled

        if self.n_labeled > current_index + self.batch_size:
            current_batch_size = self.batch_size
            self.batch_index += 1
        else:
            current_batch_size = self.n_labeled - current_index
            self.batch_index = 0

        self.global_step_index += 1

        batch = self._material_samples_flattened[current_index: current_index + current_batch_size]
        return batch, current_index, current_batch_size, self.global_step_index

    @property
    def num_steps_per_epoch(self):
        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            return self._get_number_of_batches(self._num_unique_material_samples, self.batch_size)
        # If all classes are sampled uniformly, we have been through all the samples in the data
        # On average after we have gone through all the samples in the largest class, but min and mean are also valid
        else:
            return self._get_number_of_batches(self._samples_per_material_category_per_epoch * self._num_non_zero_classes, self.batch_size)
