# coding=utf-8

"""
This file is used to store project-wide global values.
"""
VALIDATION_DATA_GENERATOR_WORKERS = 2
TRAINING_DATA_GENERATOR_WORKERS = 3
DATA_GENERATION_THREADS_PER_PROCESS = 4
VALIDATION_DATA_MAX_QUEUE_SIZE = 10
TRAINING_DATA_MAX_QUEUE_SIZE = 10

USE_MULTIPROCESSING = True

MAX_NUMBER_OF_JOBS = 32
LOG_RUSAGE = False
LOG_RUSAGE_INTERVAL = 2000  # How many steps between RUSAGE logs

# Enables various debug prints and saving of batch images etc.
DEBUG = False

# Enables profiling information to be shown
PROFILE = False

DEFAULT_IMAGE_DATA_FORMAT = 'channels_last'
DEFAULT_NUMPY_FLOAT_DTYPE = 'float32'

# Override training/validation length e.g. for profiling or debugging
OVERRIDE_STEPS = True
OVERRIDE_TRAINING_STEPS_PER_EPOCH = 30
OVERRIDE_VALIDATION_STEPS_PER_EPOCH = 10
OVERRIDE_NUM_EPOCHS = 3
