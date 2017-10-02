"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import print_function

import multiprocessing
import random
import threading
import time
from abc import abstractmethod
from multiprocessing.pool import ThreadPool

import numpy as np

from keras.utils.data_utils import SequenceEnqueuer

try:
    import queue
except ImportError:
    import Queue as queue


class Sequence(object):
    @abstractmethod
    def get_batch(self, e_idx, b_idx):
        # type: (int, int) -> (list, list)

        """Gets a specific batch from a specific epoch.

        # Arguments
            :param e_idx: epoch index
            :param b_idx: batch index

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        raise NotImplementedError


def get_index(ds, e_idx, b_idx):
    # type: (Sequence, int, int) -> (list, list)

    """Quick fix for Python2, otherwise, it cannot be pickled.

    # Arguments
        ds: a Sequence object
        e_idx: epoch index
        b_idx: batch index

    # Returns
        The value at index `i`.
    """
    return ds.get_batch(e_idx=e_idx, b_idx=b_idx)


class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, sequence,
                 use_multiprocessing=False,
                 shuffle=False,
                 initial_epoch=0,
                 max_epoch=None):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing
        self.shuffle = shuffle
        self.workers = 0
        self.executor = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None
        self.e_idx = initial_epoch
        self.max_epoch = max_epoch

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor = multiprocessing.Pool(workers)
        else:
            self.executor = ThreadPool(workers)
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        """Function to submit request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))

        while True:
            # Prevent useless epochs from running
            if self.max_epoch is not None:
                if self.e_idx >= self.max_epoch:
                    break

            if self.shuffle:
                random.shuffle(sequence)

            for b_idx in sequence:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    self.executor.apply_async(get_index,
                                              (self.sequence, self.e_idx, b_idx)), block=True)
            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self.e_idx += 1

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            raise StopIteration(e)

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.executor.close()
        self.executor.join()
        self.run_thread.join(timeout)


class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
