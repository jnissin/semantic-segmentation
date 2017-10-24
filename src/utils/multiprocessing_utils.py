# coding=utf-8

import threading
import multiprocessing

from src import settings

class ThreadsafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))

    return g


class MultiprocessingPool(object):
    def __init__(self, num_workers):
        # type: (int) -> None

        self._num_workers = num_workers
        self._pool = multiprocessing.Pool(num_workers)
        self._used = False

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def pool(self):
        return self._pool

    @property
    def used(self):
        return self._used

    @used.setter
    def used(self, val):
        self._used = val


class MultiprocessingManager(object):
    def __init__(self):
        # type: () -> None
        self._manager = multiprocessing.Manager()
        self._used = False

    @property
    def manager(self):
        return self._manager

    @property
    def used(self):
        return self._used

    @used.setter
    def used(self, val):
        self._used = val


_CACHED_POOL_SIZES = [settings.TRAINING_DATA_GENERATOR_WORKERS, settings.VALIDATION_DATA_GENERATOR_WORKERS, settings.VALIDATION_DATA_GENERATOR_WORKERS]
_POOL_CACHE = []
_MANAGER_CACHE = []


def initialize_multiprocessing_pool_cache():
    if settings.USE_MULTIPROCESSING:
        global _WORKERS_IN_POOL
        global _USED_POOLS
        global _POOL_CACHE

        for val in _CACHED_POOL_SIZES:
            _POOL_CACHE.append(MultiprocessingPool(num_workers=val))


def get_cached_multiprocessing_pool(num_workers):
    # type: (int) -> multiprocessing.Pool

    if settings.USE_MULTIPROCESSING:
        global _POOL_CACHE

        for pool in _POOL_CACHE:
            if not pool.used and pool.num_workers == num_workers:
                pool.used = True
                return pool.pool

    # No cached Pools available or not using multiprocessing
    return None


def initialize_multiprocessing_manager_cache():
    if settings.USE_MULTIPROCESSING:
        global _CACHED_POOL_SIZES
        global _MANAGER_CACHE

        for _ in _CACHED_POOL_SIZES:
            _MANAGER_CACHE.append(MultiprocessingManager())


def get_cached_multiprocessing_manager():
    if settings.USE_MULTIPROCESSING:
        global _MANAGER_CACHE

        for manager in _MANAGER_CACHE:
            if not manager.used:
                manager.used = True
                return manager.manager

    # No cached Managers available or not using multiprocessing
    return None
