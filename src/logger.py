# coding=utf-8

import os
import datetime
from enum import Enum

import settings
from utils import general_utils
from keras.preprocessing.image import array_to_img


class LogLevel(Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    PROFILE = 'PROFILE'
    DEBUG = 'DEBUG'


class Logger(object):

    def __init__(self, log_file_path, log_images_folder_path=None, use_timestamp=True, log_to_stdout_default=True):
        # type: (str, bool, bool) -> None

        self.log_file_path = log_file_path
        self.log_file = None
        self.use_timestamp = use_timestamp
        self.log_to_stdout_default = log_to_stdout_default

        # Create log images folder if it doesn't exist
        if log_images_folder_path is not None:
            self.log_images_folder_path = log_images_folder_path
        else:
            self.log_images_folder_path = os.path.join(os.path.dirname(log_file_path), 'log_images/')

        general_utils.create_path_if_not_existing(self.log_images_folder_path)

    @property
    def log_folder_path(self):
        return os.path.dirname(self.log_file_path)

    def log(self, message, log_level=LogLevel.INFO, log_to_stdout=False):
        # type: (str, LogLevel, bool) -> None

        if log_level == LogLevel.DEBUG and not settings.DEBUG or \
           log_level == LogLevel.PROFILE and not settings.PROFILE:
            return

        log_to_stdout = log_to_stdout or self.log_to_stdout_default

        # If log file is not open - open
        if not self.log_file:
            self.open_log()

        # Add timestamp
        if self.use_timestamp:
            message = '{:%Y-%m-%d %H:%M:%S}: {}'.format(datetime.datetime.now(), message)

        # Add log level
        message = '{} {}'.format(log_level.value, message)

        # Log to file - make sure there is a newline
        if not message.endswith('\n'):
            self.log_file.write(message + '\n')
        else:
            self.log_file.write(message)

        # Log to stdout - no newline needed
        if log_to_stdout:
            print message.strip()

    def warn(self, message, log_to_stdout=False):
        self.log(message, log_level=LogLevel.WARNING, log_to_stdout=log_to_stdout)

    def log_image(self, np_image, file_name, scale=True, format='JPEG'):
        image = array_to_img(np_image, scale=scale)
        image.save(os.path.join(self.log_images_folder_path, file_name), format=format)

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
        if not self.log_file:
            if self.log_file_path:
                general_utils.create_path_if_not_existing(self.log_file_path)
                mode = 'a' if os.path.exists(self.log_file_path) else 'w'
                self.log_file = open(self.log_file_path, mode=mode)
            else:
                raise ValueError('Invalid log file path, cannot log')

    def close_log(self):
        # If the log file is open close it
        if self.log_file:
            self.log_file.close()
