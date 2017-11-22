"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import print_function

import multiprocessing
import random
import threading
import time
import os

from abc import abstractmethod
from multiprocessing.pool import ThreadPool
from socket import error as socket_error

import numpy as np

try:
    import queue
except ImportError:
    import Queue as queue

from src.logger import Logger
from src.utils.multiprocessing_utils import MultiprocessingManager
from src import settings


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


# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
_SHARED_DICTS = {}


def _initialize_globals(uuid):
    """Initialize the inner dictionary to manage processes."""
    global _SHARED_DICTS
    _SHARED_DICTS[uuid] = MultiprocessingManager.instance().get_shared_dict_for_uuid(uuid)


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


def _update_sequence(uuid, seq):
    """Update current process with a new Sequence.
    # Arguments
        seq: Sequence object
    """
    global _SHARED_SEQUENCES, _SHARED_DICTS
    if not multiprocessing.current_process().pid in _SHARED_DICTS[uuid]:
        _SHARED_SEQUENCES[uuid] = seq
        _SHARED_DICTS[uuid][multiprocessing.current_process().pid] = 0


def _process_init(uuid):
    # type: (int) -> None

    pid = os.getpid()
    is_daemon = multiprocessing.current_process().daemon
    Logger.instance().log('Hello from process: {} for uuid: {}, daemon: {}'.format(pid, uuid, is_daemon))

    # Clear any keras sessions from data generation child processes - they are unnecessary
    # try:
    #     from keras import backend as K
    #     Logger.instance().log('Clearing Tensorflow session from child process: {}'.format(pid))
    #     K.clear_session()
    #     memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #     Logger.instance().log('Running garbage collection from child process: {}, with memory usage: {}'.format(pid, memory_used))
    #     collected = gc.collect()
    #     memory_used_after_gc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #     Logger.instance().log('Collected {} objects from child process: {}, memory usage diff: {}'.format(collected, pid, memory_used-memory_used_after_gc))
    # except Exception as e:
    #     Logger.instance().warn('Caught exception while clearing Tensorflow session from child process {}: {}'.format(pid, e.message))


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
                 random_seed=None):
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
        self._logger = None
        self.paused = False
        self.pause_sleep_time = 1.00
        self.last_queue_size_report_time = 0.0

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Assign a unique id
        self.uuid = MultiprocessingManager.instance().get_new_client_uuid()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Logger.instance()
        return self._logger

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
            _initialize_globals(self.uuid)
            self.executor = multiprocessing.Pool(workers, _process_init, (self.uuid,))
        else:
            self.executor = ThreadPool(workers)

        self.workers = workers
        self.last_queue_size_report_time = time.time()
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        """Function to submit request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence

        while True:
            if self.paused:
                self.pause_sleep(self.pause_sleep_time)
            else:
                # Prevent useless epochs from running
                if self.max_epoch is not None:
                    if self.e_idx >= self.max_epoch:
                        break

                if self.shuffle:
                    random.shuffle(sequence)

                for b_idx in sequence:
                    if self.stop_signal.is_set():
                        return

                    self.queue.put(self.executor.apply_async(get_index, (self.uuid, self.e_idx, b_idx)), block=True)

                    if settings.QUEUE_SIZE_REPORT_INTERVAL is not None:
                        if time.time() - self.last_queue_size_report_time > settings.QUEUE_SIZE_REPORT_INTERVAL:
                            self.last_queue_size_report_time = time.time()
                            self.logger.log('Queue size: {}'.format(self.queue.qsize()))

                while not self.queue.empty():
                    pass  # Wait for the last few batches to be processed

                self.sequence.on_epoch_end()    # Call the internal on epoch end.
                self._send_sequence()           # Update the pool
                self.e_idx += 1                 # Increase the internal epoch index

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
            self.logger.warn('Caught exception while generating batch - stopping iteration: {}'.format(e.message))
            self.stop()
            raise StopIteration(e)

    def _send_sequence(self):
        """Send current Sequence to all workers."""
        global _SHARED_SEQUENCES, _SHARED_DICTS

        _SHARED_SEQUENCES[self.uuid] = self.sequence  # For new processes that may spawn

        if not self.use_multiprocessing:
            # Threads are from the same process so they already share the sequence.
            return

        self.clear_shared_dict()

        while len(_SHARED_DICTS[self.uuid]) < self.workers and not self.stop_signal.is_set():
            try:
                # Ask the pool to update till everyone is updated.
                self.executor.apply(_update_sequence, args=(self.uuid, self.sequence,))
            except socket_error as e:
                self.logger.warn('Failed to update sequence: {}'.format(e.strerror))
                break

        # We're done with the update

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

        # Clean up any resources shared by the processes
        global _SHARED_DICTS, _SHARED_SEQUENCES
        _SHARED_SEQUENCES[self.uuid] = None

        if self.use_multiprocessing:
            if _SHARED_DICTS.get(self.uuid) is not None:
                self.clear_shared_dict()
                _SHARED_DICTS[self.uuid] = None

    def continue_run(self):
        self.paused = False

    def pause_run(self):
        self.paused = True

    def clear_shared_dict(self):
        """
        Clears the shared dict with the uuid associated with this OrderedEnqueuer
        uuid.
        """

        try:
            if self.uuid in _SHARED_DICTS:
                if _SHARED_DICTS.get(self.uuid) is not None:
                    _SHARED_DICTS[self.uuid].clear()
        except socket_error as e:
            self.logger.warn('Failed to clear _SHARED_DICTS: {}'.format(e.strerror))


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
        self.paused = False
        self.pause_sleep_time = 1.0
        self.last_queue_size_report_time = 0.0

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def start(self, workers=1, max_queue_size=10, start_paused=False):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        # Initialize the pause state
        self.paused = start_paused

        def data_generator_task():
            while not self._stop_event.is_set():
                if self.paused:
                    self.pause_sleep(self.pause_sleep_time)
                else:
                    try:
                        if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                            generator_output = next(self._generator)
                            self.queue.put(generator_output)
                        else:
                            time.sleep(self.wait_time)

                        if settings.QUEUE_SIZE_REPORT_INTERVAL is not None:
                            if time.time() - self.last_queue_size_report_time > settings.QUEUE_SIZE_REPORT_INTERVAL:
                                self.last_queue_size_report_time = time.time()
                                self.logger.log('Queue size: {}'.format(self.queue.qsize()))

                    except Exception:
                        self._stop_event.set()
                        raise

        try:
            self.last_queue_size_report_time = time.time()

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

    def continue_run(self):
        self.paused = False

    def pause_run(self):
        self.paused = True
