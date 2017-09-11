# coding=utf-8

import math
import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum

from logger import Logger
from utils.dataset_utils import MaterialSample


class DataSetIterator(object):

    __metaclass__ = ABCMeta

    def __init__(self, n, batch_size, shuffle, seed, logger):
        self.n = n
        self.batch_size = min(batch_size, n)    # The batch size could in theory be bigger than the data set size
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger
        self.batch_index = 0                    # The index of the batch within the epoch
        self.step_index = 0                     # The overall index of the batch

    def reset(self):
        self.batch_index = 0
        self.step_index = 0

    @abstractmethod
    def get_next_batch(self):
        if self.seed is not None:
            np.random.seed(self.seed + self.step_index)

    @abstractproperty
    def num_steps_per_epoch(self):
        pass

    def _get_number_of_batches(self, data_set_size, batch_size):
        # type: (int, int) -> int

        """
        Returns the number of batches for the given data set size and batch size.
        The function assumes that all data will be used every epoch and the last batch size
        can be smaller than the others.

        # Arguments
            :param data_set_size: data set size
            :param batch_size: batch size
        # Returns
            :return: the number of batches from this dataset
        """
        num_batches = int(math.ceil(float(data_set_size) / float(batch_size)))
        return num_batches


class BasicDataSetIterator(DataSetIterator):
    """
    A class for iterating over a data set in batches.
    """

    def __init__(self, n, batch_size, shuffle, seed, logger):
        # type: (int, int, bool, int, Logger) -> None

        """
        # Arguments
            :param n: Integer, total number of samples in the dataset to loop over.
            :param batch_size: Integer, size of a batch.
            :param shuffle: Boolean, whether to shuffle the data between epochs.
            :param seed: Random seeding for data shuffling.
            :param logger: Logger instance for logging
        # Returns
            Nothing
        """
        super(BasicDataSetIterator, self).__init__(n=n, batch_size=batch_size, shuffle=shuffle, seed=seed, logger=logger)
        self.index_array = np.arange(self.n)

    def get_next_batch(self):
        # type: () -> (np.ndarray[int], int, int)

        super(BasicDataSetIterator, self).get_next_batch()

        if self.batch_index == 0:
            self.index_array = np.arange(self.n)
            if self.shuffle:
                self.index_array = np.random.permutation(self.n)

        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:
            current_batch_size = self.batch_size
            self.batch_index += 1
        else:
            current_batch_size = self.n - current_index
            self.batch_index = 0

        self.step_index += 1

        return self.index_array[current_index: current_index + current_batch_size], current_index, current_batch_size, self.step_index

    @property
    def num_steps_per_epoch(self):
        return self._get_number_of_batches(self.n, self.batch_size)


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

    def get_next_batch(self):
        # type: () -> (list[tuple[int]], int, int)

        """
        Gives the next batch as tuple of indices to the material samples 2D array.
        The tuples are in the form (sample_category_idx, sample_idx).

        # Arguments
            Nothing
        # Returns
            :return: a batch of material samples as tuples of indices into 2D material sample array
        """

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

        self.step_index += 1

        return batch, current_index, current_batch_size, self.step_index

    def _get_next_batch_unique(self):
        if self.batch_index == 0 and self.shuffle:
            np.random.shuffle(self._material_samples_flattened)

        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:
            current_batch_size = self.batch_size
            self.batch_index += 1
        else:
            current_batch_size = self.n - current_index
            self.batch_index = 0

        self.step_index += 1

        batch = self._material_samples_flattened[current_index: current_index + current_batch_size]
        return batch, current_index, current_batch_size, self.step_index

    @property
    def num_steps_per_epoch(self):
        if self.iter_mode == MaterialSampleIterationMode.UNIQUE:
            return self._get_number_of_batches(self._num_unique_material_samples, self.batch_size)
        # If all classes are sampled uniformly, we have been through all the samples in the data
        # On average after we have gone through all the samples in the largest class, but min and mean are also valid
        else:
            return self._get_number_of_batches(self._samples_per_material_category_per_epoch * self._num_non_zero_classes, self.batch_size)
