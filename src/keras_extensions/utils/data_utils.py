"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import print_function

import multiprocessing
import random
import threading
import time
import os
import six
import sys
import traceback

from abc import abstractmethod
from multiprocessing.pool import ThreadPool
from contextlib import closing

import numpy as np

try:
    import queue
except ImportError:
    import Queue as queue

from src.logger import Logger
from src import settings


###############################################
# GLOBALS
###############################################

# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


def get_index(uuid, e_idx, b_idx):
    # type: (int, int) -> (list, list)

    """Quick fix for Python2, otherwise, it cannot be pickled.

    # Arguments
        ds: a Sequence object
        e_idx: epoch index
        b_idx: batch index

    # Returns
        The value at index `i`.
    """
    global _SHARED_SEQUENCES
    return _SHARED_SEQUENCES[uuid].get_batch(e_idx=e_idx, b_idx=b_idx)


def init_pool(uuid, seed, seqs):
    # type: (int, int, dict) -> None
    global _SHARED_SEQUENCES

    # Print process information
    pid = os.getpid()
    is_daemon = multiprocessing.current_process().daemon
    Logger.instance().log('Hello from process: {} for uuid: {}, daemon: {}'.format(pid, uuid, is_daemon))

    # Initialize the random seed
    random.seed(seed)
    np.random.seed(seed)

    # Set the sequence
    _SHARED_SEQUENCES = seqs


###############################################
# SEQUENCE
###############################################


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


class SequenceEnqueuer(object):
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    # Examples

    ```python
    enqueuer = SequenceEnqueuer(...)
    enqueuer.start()
    datas = enqueuer.get()
    for data in datas:
        # Use the inputs; training, evaluating, predicting.
        # ... stop sometime.
    enqueuer.close()
    ```

    The `enqueuer.get()` should be an infinite stream of datas.

    """

    @abstractmethod
    def is_running(self):
        raise NotImplementedError

    @abstractmethod
    def start(self, workers=1, max_queue_size=10, start_paused=False):
        """Starts the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`).
            start_paused: should the initial state be paused?
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError

    @abstractmethod
    def continue_run(self):
        raise NotImplementedError

    @abstractmethod
    def pause_run(self):
        raise NotImplementedError

    def pause_sleep(self, pause_sleep_time):
        try:
            time.sleep(pause_sleep_time)
        except (NameError, AttributeError, ValueError):
            import time
            time.sleep(pause_sleep_time)


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
                 max_epoch=None,
                 seed=None):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.shuffle = shuffle
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None
        self.e_idx = initial_epoch
        self.max_epoch = max_epoch
        self._logger = None
        self.paused = False
        self.pause_sleep_time = 1.00
        self.last_queue_size_report_time = 0.0
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10, start_paused=False):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        # Initialize the pause state
        self.paused = start_paused

        if self.use_multiprocessing:
            self.executor_fn = lambda seqs: multiprocessing.Pool(workers,
                                                                 initializer=init_pool,
                                                                 initargs=(self.uid, self.seed, seqs,))
        else:
            # We do not need the init since it's threads.
            self.executor_fn = ThreadPool(workers)

        self.workers = workers
        self.last_queue_size_report_time = time.time()
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        """Function to submit request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence

        while True:
            # Prevent useless epochs from running
            if self.max_epoch is not None:
                if self.e_idx >= self.max_epoch:
                    break

            if not self.paused:
                if self.shuffle:
                    random.shuffle(sequence)

                with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:

                    for b_idx in sequence:
                        if self.stop_signal.is_set():
                            return

                        self.queue.put(
                            executor.apply_async(get_index, (self.uid, self.e_idx, b_idx)), block=True)

                        if settings.QUEUE_SIZE_REPORT_INTERVAL is not None:
                            if time.time() - self.last_queue_size_report_time > settings.QUEUE_SIZE_REPORT_INTERVAL:
                                self.last_queue_size_report_time = time.time()
                                self.logger.log('Queue size: {}'.format(self.queue.qsize()))

                    # Done with the current epoch, waiting for the final batches
                    self._wait_queue()

                    if self.stop_signal.is_set():
                        # We're done
                        return

                # Call
                self.sequence.on_epoch_end()    # Call the internal on epoch end.
                self._send_sequence()           # Update the pool
                self.e_idx += 1                 # Increase the internal epoch index
            else:
                self.pause_sleep(self.pause_sleep_time)
                continue

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
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.logger.warn('Caught exception while generating batch - stopping iteration: {}'.format(e.message))
            self.stop()
            six.raise_from(StopIteration(e), e)

    def _send_sequence(self):
        """Send current Sequence to all workers."""
        global _SHARED_SEQUENCES
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

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

        self.run_thread.join(timeout)

        # Clean up any resources shared by the processes
        global _SHARED_SEQUENCES
        _SHARED_SEQUENCES[self.uid] = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Logger.instance()
        return self._logger

    def continue_run(self):
        self.paused = False

    def pause_run(self):
        self.paused = True


class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 seed=None):
        self.wait_time = wait_time
        self._generator = generator
        if os.name is 'nt' and use_multiprocessing is True:
            # On Windows, avoid **SYSTEMATIC** error in `multiprocessing`:
            # `TypeError: can't pickle generator objects`
            # => Suggest multithreading instead of multiprocessing on Windows
            raise ValueError('Using a generator with `use_multiprocessing=True`'
                             ' is not supported on Windows (no marshalling of'
                             ' generators across process boundaries). Instead,'
                             ' use single thread/process or multithreading.')
        else:
            self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self._manager = None
        self.queue = None
        self.seed = seed
        self.paused = False
        self.pause_sleep_time = 1.0
        self.last_queue_size_report_time = 0.0
        self._logger = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _data_generator_task(self):
        if self._use_multiprocessing is False:
            while not self._stop_event.is_set():
                if self.paused:
                    self.pause_sleep(self.pause_sleep_time)
                else:
                    with self.genlock:
                        try:
                            if (self.queue is not None and
                                    self.queue.qsize() < self.max_queue_size):
                                # On all OSes, avoid **SYSTEMATIC** error
                                # in multithreading mode:
                                # `ValueError: generator already executing`
                                # => Serialize calls to
                                # infinite iterator/generator's next() function
                                generator_output = next(self._generator)
                                self.queue.put((True, generator_output))
                            else:
                                time.sleep(self.wait_time)

                            if settings.QUEUE_SIZE_REPORT_INTERVAL is not None:
                                if time.time() - self.last_queue_size_report_time > settings.QUEUE_SIZE_REPORT_INTERVAL:
                                    self.last_queue_size_report_time = time.time()
                                    self.logger.log('Queue size: {}'.format(self.queue.qsize()))
                        except StopIteration:
                            break
                        except Exception as e:
                            # Can't pickle tracebacks.
                            # As a compromise, print the traceback and pickle None instead.
                            if not hasattr(e, '__traceback__'):
                                setattr(e, '__traceback__', sys.exc_info()[2])
                            self.queue.put((False, e))
                            self._stop_event.set()
                            break
        else:
            while not self._stop_event.is_set():
                if self.paused:
                    self.pause_sleep(self.pause_sleep_time)
                else:
                    try:
                        if (self.queue is not None and
                                self.queue.qsize() < self.max_queue_size):
                            generator_output = next(self._generator)
                            self.queue.put((True, generator_output))
                        else:
                            time.sleep(self.wait_time)

                        if settings.QUEUE_SIZE_REPORT_INTERVAL is not None:
                            if time.time() - self.last_queue_size_report_time > settings.QUEUE_SIZE_REPORT_INTERVAL:
                                self.last_queue_size_report_time = time.time()
                                self.logger.log('Queue size: {}'.format(self.queue.qsize()))
                    except StopIteration:
                        break
                    except Exception as e:
                        # Can't pickle tracebacks.
                        # As a compromise, print the traceback and pickle None instead.
                        traceback.print_exc()
                        setattr(e, '__traceback__', None)
                        self.queue.put((False, e))
                        self._stop_event.set()
                        break

    def start(self, workers=1, max_queue_size=10, start_paused=False):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """
        try:
            self.max_queue_size = max_queue_size
            if self._use_multiprocessing:
                self._manager = multiprocessing.Manager()
                self.queue = self._manager.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                # On all OSes, avoid **SYSTEMATIC** error in multithreading mode:
                # `ValueError: generator already executing`
                # => Serialize calls to infinite iterator/generator's next() function
                self.genlock = threading.Lock()
                self.queue = queue.Queue(maxsize=max_queue_size)
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.seed)
                    thread = multiprocessing.Process(target=self._data_generator_task)
                    thread.daemon = True
                    if self.seed is not None:
                        self.seed += 1
                else:
                    thread = threading.Thread(target=self._data_generator_task)
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
            if self._use_multiprocessing:
                if thread.is_alive():
                    thread.terminate()
            else:
                # The thread.is_alive() test is subject to a race condition:
                # the thread could terminate right after the test and before the
                # join, rendering this test meaningless -> Call thread.join()
                # always, which is ok no matter what the status of the thread.
                thread.join(timeout)

        if self._manager:
            self._manager.shutdown()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            if not self.queue.empty():
                success, value = self.queue.get()
                # Rethrow any exceptions found in the queue
                if not success:
                    six.reraise(value.__class__, value, value.__traceback__)
                # Yield regular values
                if value is not None:
                    yield value
            else:
                all_finished = all([not thread.is_alive() for thread in self._threads])
                if all_finished and self.queue.empty():
                    raise StopIteration()
                else:
                    time.sleep(self.wait_time)

        # Make sure to rethrow the first exception in the queue, if any
        while not self.queue.empty():
            success, value = self.queue.get()
            if not success:
                six.reraise(value.__class__, value, value.__traceback__)

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Logger.instance()
        return self._logger

    def continue_run(self):
        self.paused = False

    def pause_run(self):
        self.paused = True
