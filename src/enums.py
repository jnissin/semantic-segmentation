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
    FELZENSWALB = 1
    SLIC = 2
    WATERSHED = 3
    QUICKSHIFT = 4
