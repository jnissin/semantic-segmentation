# coding=utf-8

import os
import datetime
import numpy as np

from PIL import Image as PImage
from multiprocessing import Lock
from enum import Enum

from utils.image_utils import array_to_img
from utils import general_utils

import settings


class LogLevel(Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    PROFILE = 'PROFILE'
    DEBUG = 'DEBUG'


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(object):

    __metaclass__ = Singleton

    _instance = None
    _file_write_lock = Lock()
    _log_images_folder_path = None
    _log_file_path = None
    _log_file = None
    _buf_size = 5

    def __init__(self, log_file_path, log_images_folder_path=None, use_timestamp=True, log_to_stdout_default=True, stdout_only=False):
        # type: (str, bool, bool) -> None
        self.use_timestamp = use_timestamp
        self.log_to_stdout_default = log_to_stdout_default
        self.stdout_only = stdout_only

        Logger._log_images_folder_path = log_images_folder_path
        Logger._log_file_path = log_file_path

        Logger._log_file = None
        Logger._instance = self

        # Create log images folder if it doesn't exist
        if log_images_folder_path is not None:
            Logger._log_images_folder_path = log_images_folder_path
        else:
            if not self.stdout_only:
                Logger._log_images_folder_path = os.path.join(os.path.dirname(log_file_path), 'log_images/')

        if not self.stdout_only and Logger._log_images_folder_path is not None:
            general_utils.create_path_if_not_existing(Logger._log_images_folder_path)

    @staticmethod
    def instance():
        # type: () -> Logger
        if Logger._instance is None:
            raise ValueError('Logger has not been initialized')

        return Logger._instance

    @staticmethod
    def format_message(message, log_level=LogLevel.INFO, use_timestamp=True):
        # Add timestamp
        if use_timestamp:
            message = '{:%Y-%m-%d %H:%M:%S}: {}'.format(datetime.datetime.now(), message)

        # Add log level
        message = '{} {}'.format(log_level.value, message)
        return message

    @staticmethod
    def log_file_is_open():
        # type: () -> bool
        return Logger._log_file is not None and not Logger._log_file.closed

    @property
    def log_folder_path(self):
        return os.path.dirname(Logger._log_file_path)

    def log(self, message, log_level=LogLevel.INFO, log_to_stdout=False):
        # type: (str, LogLevel, bool) -> None

        if log_level == LogLevel.DEBUG and not settings.DEBUG or \
           log_level == LogLevel.PROFILE and not settings.PROFILE:
            return

        message = Logger.format_message(message, log_level=log_level, use_timestamp=self.use_timestamp)

        # Log to file - make sure there is a newline
        if not self.stdout_only:
            # If log file is not open - open
            if not Logger.log_file_is_open():
                self.open_log()

            with Logger._file_write_lock:
                if not message.endswith('\n'):
                    Logger._log_file.write(message + '\n')
                else:
                    Logger._log_file.write(message)

        # Log to stdout - no newline needed
        if log_to_stdout or self.log_to_stdout_default or self.stdout_only:
            print message.strip()

    def warn(self, message, log_to_stdout=False):
        self.log(message, log_level=LogLevel.WARNING, log_to_stdout=log_to_stdout)

    def log_image(self, img, file_name, scale=True, format='JPEG'):
        if Logger._log_images_folder_path is not None:
            if isinstance(img, np.ndarray):
                image = array_to_img(img, scale=scale)
            elif isinstance(img, PImage.Image):
                image = img
            else:
                raise ValueError('Unsupported image type: {}'.format(type(img)))

            image.save(os.path.join(Logger._log_images_folder_path, file_name), format=format)
        else:
            self.log('Attempting to use log_image when log_images_folder_path is None, stdout_only: {}'.format(self.stdout_only), LogLevel.WARNING)

    def debug_log(self, message, log_to_stdout=False):
        if settings.DEBUG:
            self.log(message, log_level=LogLevel.DEBUG, log_to_stdout=log_to_stdout)

    def debug_log_image(self, np_image, file_name, scale=True, format='JPEG'):
        if settings.DEBUG:
            self.log_image(np_image, file_name, scale, format)

    def profile_log(self, message, log_to_stdout=False):
        if settings.PROFILE:
            self.log(message, log_level=LogLevel.PROFILE, log_to_stdout=log_to_stdout)

    def open_log(self):
        # Create and open the log file
        if not Logger.log_file_is_open():
            with Logger._file_write_lock:
                if Logger._log_file_path:
                        general_utils.create_path_if_not_existing(Logger._log_file_path)
                        mode = 'a' if os.path.exists(Logger._log_file_path) else 'w'
                        Logger._log_file = open(Logger._log_file_path, mode, Logger._buf_size)
                else:
                    raise ValueError('Invalid log file path, cannot log')

    def close_log(self):
        with Logger._file_write_lock:
            # If the log file is open close it
            if Logger._log_file is not None and not Logger._log_file.closed:
                Logger._log_file.close()
