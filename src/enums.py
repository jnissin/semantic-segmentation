# coding=utf-8

from enum import Enum


class ClassWeightType(Enum):
    NONE = 0
    MEDIAN_FREQUENCY_BALANCING = 1
    ENET = 2


class CoordinateType(Enum):
    ABSOLUTE = 0
    NORMALIZED = 1


class ImageType(Enum):
    NONE = 0
    PHOTO = 1
    MASK = 2


class BatchDataFormat(Enum):
    SUPERVISED = 0
    SEMI_SUPERVISED = 1


class SuperpixelSegmentationFunctionType(Enum):
    NONE = 0
    FELZENSZWALB = 1
    SLIC = 2
    WATERSHED = 3
    QUICKSHIFT = 4


class MaterialSampleIterationMode(Enum):
    NONE = -1           # None
    UNIFORM_MAX = 0     # Sample each material class uniformly. Set the number of steps per epoch according to max class num samples.
    UNIFORM_MIN = 1     # Sample each material class uniformly. Set the number of steps per epoch according to min class num samples.
    UNIFORM_MEAN = 2    # Sample each class uniformly. Set the number of steps per epoch according to mean samples per class.
    UNIQUE = 3          # Iterate through all the unique samples once within epoch - means no balancing