# coding=utf-8

from enum import Enum


class BatchDataFormat(Enum):
    SUPERVISED = 0
    SEMI_SUPERVISED = 1


class SuperpixelSegmentationFunctionType(Enum):
    NONE = 0
    FELZENSWALB = 1
    SLIC = 2
    WATERSHED = 3
    QUICKSHIFT = 4
