# coding=utf-8

import threading
import multiprocessing

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


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MultiprocessingManager(object):
    __metaclass__ = Singleton

    _MANAGER = None
    _INSTANCE = None
    _NEXT_CLIENT_UUID = None
    _NUM_CURRENT_CLIENTS = None
    _SHARED_DICTS = {}

    def __init__(self):
        MultiprocessingManager._MANAGER = multiprocessing.Manager()
        MultiprocessingManager._NEXT_CLIENT_UUID = multiprocessing.Value('i', 0)
        MultiprocessingManager._NUM_CURRENT_CLIENTS = multiprocessing.Value('i', 0)
        MultiprocessingManager._INSTANCE = self

    @staticmethod
    def instance():
        # type: () -> MultiprocessingManager
        if MultiprocessingManager._INSTANCE is None:
            MultiprocessingManager._INSTANCE = MultiprocessingManager()

        return MultiprocessingManager._INSTANCE

    @property
    def num_current_clients(self):
        # type: () -> int
        return MultiprocessingManager._NUM_CURRENT_CLIENTS.value

    @property
    def manager(self):
        # type: () -> multiprocessing.Manager
        return MultiprocessingManager._MANAGER

    def get_new_client_uuid(self):
        # type: () -> int
        uuid = MultiprocessingManager._NEXT_CLIENT_UUID.value
        MultiprocessingManager._NEXT_CLIENT_UUID.value += 1
        MultiprocessingManager._NUM_CURRENT_CLIENTS.value += 1
        return uuid

    def get_shared_dict_for_uuid(self, client_uuid):
        # type: () -> dict

        if client_uuid not in MultiprocessingManager._SHARED_DICTS:
            MultiprocessingManager._SHARED_DICTS[client_uuid] = MultiprocessingManager._MANAGER.dict()

        return MultiprocessingManager._SHARED_DICTS[client_uuid]
