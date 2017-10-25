# coding=utf-8

import threading
import multiprocessing

from src import settings


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


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


class ManagedMultiprocessingPool(object):
    def __init__(self, pool_uuid, num_workers):
        # type: (int) -> None

        self._pool_uuid = pool_uuid
        self._num_workers = num_workers
        self._pool = multiprocessing.Pool(num_workers)
        self._creator_pid = multiprocessing.current_process().pid
        self._assigned_client_uuid = None

    @property
    def uuid(self):
        # type: () -> int
        return self._pool_uuid

    @property
    def num_workers(self):
        # type: () -> int
        return self._num_workers

    @property
    def value(self):
        # type: () -> multiprocessing.Pool
        return self._pool

    @property
    def used(self):
        # type: () -> bool
        return self._assigned_client_uuid is not None

    @property
    def creator_pid(self):
        # type: () -> int
        return self._creator_pid

    @property
    def client_uuid(self):
        # type: () -> int
        return self._assigned_client_uuid

    def assign_to_client(self, client_uuid):
        self._assigned_client_uuid = client_uuid

    def release(self):
        self._assigned_client_uuid = None


class MultiprocessingManager(object):
    __metaclass__ = Singleton

    _INSTANCE = None
    _MANAGER = None
    _SHARED_DICTS = None
    _SHARED_OBJECTS = None

    _CACHED_POOL_SIZES = [settings.TRAINING_DATA_GENERATOR_WORKERS, settings.VALIDATION_DATA_GENERATOR_WORKERS, settings.VALIDATION_DATA_GENERATOR_WORKERS]
    _POOL_CACHE = []
    _POOL_UUID = multiprocessing.Value('i', 0)
    _CLIENT_UUID = multiprocessing.Value('i', 0)
    _OBJECT_UUID = multiprocessing.Value('i', 0)

    def __init__(self):
        # type: () -> None
        MultiprocessingManager._INSTANCE = self
        MultiprocessingManager._MANAGER = multiprocessing.Manager()
        MultiprocessingManager._SHARED_DICTS = MultiprocessingManager._MANAGER.dict()
        MultiprocessingManager._SHARED_OBJECTS = MultiprocessingManager._MANAGER.dict()

        # Initialize the processing pool cache
        for val in MultiprocessingManager._CACHED_POOL_SIZES:
            self._add_new_process_pool(val)

    @staticmethod
    def instance():
        # type: () -> MultiprocessingManager

        if MultiprocessingManager._INSTANCE is None:
            MultiprocessingManager._INSTANCE = MultiprocessingManager()
        return MultiprocessingManager._INSTANCE

    @property
    def manager(self):
        # type: () -> multiprocessing.Manager

        return MultiprocessingManager._MANAGER

    def get_next_client_uuid(self):
        # type: () -> int

        uuid = MultiprocessingManager._CLIENT_UUID.value
        MultiprocessingManager._CLIENT_UUID.value += 1
        return uuid

    def get_process_pool_for_client(self, num_workers, client_uuid):
        # type: (int) -> multiprocessing.Pool

        for pool in MultiprocessingManager._POOL_CACHE:
            if not pool.used and pool.num_workers == num_workers:
                pool.assign_to_client(client_uuid)
                return pool.value

        # Create a new multiprocessing pool
        pool = self._add_new_process_pool(num_workers)
        pool.assign_to_client(client_uuid)

        return pool.value

    def release_client_process_pools(self, client_uuid):
        # type: (int) -> None

        for pool in MultiprocessingManager._POOL_CACHE:
            if pool.client_uuid == client_uuid:
                pool.release()

    def get_shared_dict_for_client(self, client_uuid):
        # type: (int) -> dict

        if client_uuid not in MultiprocessingManager._SHARED_DICTS:
            MultiprocessingManager._SHARED_DICTS[client_uuid] = {}

        return MultiprocessingManager._SHARED_DICTS[client_uuid]

    def clear_shared_dict_for_client(self, client_uuid):
        # type: (int) -> None

        if client_uuid in MultiprocessingManager._SHARED_DICTS:
            MultiprocessingManager._SHARED_DICTS[client_uuid].clear()

    def _add_new_process_pool(self, num_workers):
        # type: (int) -> ManagedMultiprocessingPool

        pool_uuid = MultiprocessingManager._POOL_UUID.value
        MultiprocessingManager._POOL_UUID.value += 1
        pool = ManagedMultiprocessingPool(pool_uuid, num_workers)
        MultiprocessingManager._POOL_CACHE.append(pool)

        return pool

    def set_shared_object_for_client(self, client_uuid, obj):
        # type: (int, object) -> None
        MultiprocessingManager._SHARED_OBJECTS[client_uuid] = obj

    def get_shared_object_for_client(self, client_uuid):
        # type: (int) -> object
        return MultiprocessingManager._SHARED_OBJECTS.get(client_uuid)

    def has_shared_object_for_client(self, client_uuid):
        # type: (int) -> bool
        return client_uuid in MultiprocessingManager._SHARED_OBJECTS and MultiprocessingManager._SHARED_OBJECTS.get(client_uuid) is not None
