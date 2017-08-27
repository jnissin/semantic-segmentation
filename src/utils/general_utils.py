# coding=utf-8

import os
import warnings

from keras.preprocessing.image import array_to_img
from tensorflow.python.client import device_lib

from .. import settings


def warn(message, category=UserWarning):
    warnings.warn(message, category, stacklevel=2)


def create_path_if_not_existing(path):
    if not path:
        return False

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        return True
    else:
        return False


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
