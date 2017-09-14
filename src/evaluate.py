# coding=utf-8

import numpy as np
import argparse
import json
import os
import time

from keras import backend as K

import losses
import metrics

from logger import Logger
from extended_optimizers import Adam, SGD, Optimizer
from data_set import LabeledImageDataSet
from generators import SegmentationDataGeneratorParameters, BatchDataFormat, SegmentationDataGenerator, ClassificationDataGenerator

from utils import dataset_utils
from models.models import get_model

# GLOBALS

CONFIG = None


def read_config_json(path):
    with open(path) as f:
        data = f.read()
        return json.loads(data)


def get_config_value(key):
    global CONFIG
    return CONFIG[key] if key in CONFIG else None


def get_latest_weights_file_path(weights_folder_path):
    weight_files = dataset_utils.get_files(weights_folder_path)

    if len(weight_files) > 0:
        weight_files.sort()
        weight_file = weight_files[-1]
        return os.path.join(weights_folder_path, weight_file)

    return None


def get_model_optimizer():
    # type: (dict) -> Optimizer

    optimizer_info = get_config_value('optimizer')
    optimizer_name = optimizer_info['name'].strip().lower()

    if optimizer_name == 'adam':
        lr = optimizer_info['learning_rate']
        decay = optimizer_info['decay']
        optimizer = Adam(lr=lr, decay=decay)

        print 'Using {} optimizer with learning rate: {}, decay: {}, beta_1: {}, beta_2: {}'\
            .format(optimizer.__class__.__name__, K.get_value(optimizer.lr), K.get_value(optimizer.decay), K.get_value(optimizer.beta_1), K.get_value(optimizer.beta_2))

    elif optimizer_name == 'sgd':
        lr = optimizer_info['learning_rate']
        decay = optimizer_info['decay']
        momentum = optimizer_info['momentum']
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay)

        print 'Using {} optimizer with learning rate: {}, momentum: {}, decay: {}'\
            .format(optimizer.__class__.__name__, K.get_value(optimizer.lr), K.get_value(optimizer.momentum), K.get_value(optimizer.decay))

    else:
        raise ValueError('Unsupported optimizer name: {}'.format(optimizer_name))

    return optimizer


def get_ignore_classes():
    ignore_classes = [0]
    return ignore_classes


def get_class_weights():
    # Unit class weights - ignore background (class 0)
    class_weights = np.ones(24, dtype=np.float32)
    class_weights[0] = 0.0

    return class_weights


def build_model(model_name, weights_path, model_type):
    # type: (str, str, str) -> Model

    # Build the model
    num_classes = 24 # TODO: Remove hard coding - move to config file?
    input_shape = get_config_value('input_shape')

    print 'Building model {} instance with input shape: {}, num classes: {}'.format(model_name, input_shape, num_classes)
    model_wrapper = get_model(model_name, input_shape, num_classes)
    model = model_wrapper.model

    # Load either provided weights or try to find the newest weights from the
    # checkpoint path
    if os.path.isdir(weights_path):
        print 'Searching for most recent weights in: {}'.format(weights_path)
        weights_path = get_latest_weights_file_path(weights_path)

    print 'Loading weights from: {}'.format(weights_path)
    model.load_weights(weights_path)

    class_weights = get_class_weights()
    ignore_classes = get_ignore_classes()

    if model_type == 'segmentation':
        model_loss_func = losses.segmentation_sparse_weighted_categorical_cross_entropy(class_weights)
        model_metrics = [metrics.segmentation_accuracy(0, ignore_classes=ignore_classes),
                         metrics.segmentation_mean_iou(num_classes, 0, ignore_classes=ignore_classes),
                         metrics.segmentation_mean_per_class_accuracy(num_classes, 0, ignore_classes=ignore_classes)]
    elif model_type == 'classification':
        model_loss_func = losses.classification_weighted_categorical_crossentropy_loss(class_weights)
        model_metrics = [metrics.classification_accuracy(0, ignore_classes=ignore_classes),
                         metrics.classification_mean_per_class_accuracy(num_classes, 0, ignore_classes=ignore_classes)]
    else:
        raise ValueError('Unknown evaluation type: {}'.format(model_type))

    optimizer = get_model_optimizer()

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=model_loss_func,
                  metrics=model_metrics)

    return model


def get_segmentation_data_generator(evaluation_type, labeled_only):
    # type: (str, bool) -> SegmentationDataGenerator

    material_class_information_path = get_config_value('path_to_material_class_file')
    path_to_data_set_information_file = get_config_value('path_to_data_set_information_file')
    path_to_labeled_photos = get_config_value('path_to_labeled_photos')
    path_to_labeled_masks = get_config_value('path_to_labeled_masks')

    print 'Loading data set information from: {}'.format(path_to_data_set_information_file)
    data_set_information = dataset_utils.load_segmentation_data_set_information(path_to_data_set_information_file)
    print 'Loaded data set information successfully with set sizes (tr, va, te): {}, {}, {}'\
        .format(data_set_information.training_set.labeled_size, data_set_information.validation_set.labeled_size, data_set_information.test_set.labeled_size)

    print 'Loading labeled {} data set with photo files from: {} and mask files from: {}'\
        .format(evaluation_type, path_to_labeled_photos, path_to_labeled_masks)

    stime = time.time()

    if evaluation_type == 'training':
        # Labeled training set
        data_set = LabeledImageDataSet('training_set_labeled',
                                       path_to_photo_archive=path_to_labeled_photos,
                                       path_to_mask_archive=path_to_labeled_masks,
                                       photo_file_list=data_set_information.training_set.labeled_photos,
                                       mask_file_list=data_set_information.training_set.labeled_masks,
                                       material_samples=data_set_information.training_set.material_samples)
    elif evaluation_type == 'validation':
        data_set = LabeledImageDataSet('validation_set',
                                       path_to_photo_archive=path_to_labeled_photos,
                                       path_to_mask_archive=path_to_labeled_masks,
                                       photo_file_list=data_set_information.validation_set.labeled_photos,
                                       mask_file_list=data_set_information.validation_set.labeled_masks,
                                       material_samples=data_set_information.validation_set.material_samples)
    elif evaluation_type == 'test':
        data_set = LabeledImageDataSet('test_set',
                                       path_to_photo_archive=path_to_labeled_photos,
                                       path_to_mask_archive=path_to_labeled_masks,
                                       photo_file_list=data_set_information.test_set.labeled_photos,
                                       mask_file_list=data_set_information.test_set.labeled_masks,
                                       material_samples=data_set_information.test_set.material_samples)

    print 'Data set ({}) creation took: {} s, size: {}'.format(evaluation_type, time.time() - stime, data_set.size)

    # Load material class information
    print 'Loading the material class information from file: {}'.format(material_class_information_path)
    material_class_information = dataset_utils.load_material_class_information(material_class_information_path)
    print 'Loaded {} material classes'.format(len(material_class_information))

    # Create data generator parameters
    num_color_channels = get_config_value('num_color_channels')
    num_labeled_per_batch = get_config_value('validation_num_labeled_per_batch')
    num_unlabeled_per_batch = 0
    random_seed = get_config_value('random_seed')
    resize_shape = get_config_value('validation_resize_shape')
    div2_constraint = get_config_value('div2_constraint')
    per_channel_mean = data_set_information.labeled_per_channel_mean if labeled_only else data_set_information.per_channel_mean
    per_channel_stddev = data_set_information.labeled_per_channel_stddev if labeled_only else data_set_information.per_channel_stddev
    class_weights = get_class_weights()

    data_generator_params = SegmentationDataGeneratorParameters(
        material_class_information=material_class_information,
        num_color_channels=num_color_channels,
        num_crop_reattempts=0,
        logger=Logger(log_file_path=None, stdout_only=True),
        random_seed=random_seed,
        crop_shapes=None,
        resize_shapes=resize_shape,
        use_per_channel_mean_normalization=True,
        per_channel_mean=per_channel_mean,
        use_per_channel_stddev_normalization=True,
        per_channel_stddev=per_channel_stddev,
        use_data_augmentation=False,
        use_material_samples=False,
        use_selective_attention=False,
        use_adaptive_sampling=False,
        data_augmentation_params=None,
        shuffle_data_after_epoch=True,
        div2_constraint=div2_constraint)

    data_generator = SegmentationDataGenerator(
        labeled_data_set=data_set,
        unlabeled_data_set=None,
        num_labeled_per_batch=num_labeled_per_batch,
        num_unlabeled_per_batch=num_unlabeled_per_batch,
        params=data_generator_params,
        class_weights=class_weights,
        batch_data_format=BatchDataFormat.SUPERVISED,
        label_generation_function=None)

    return data_generator


def get_classification_data_generator(evaluation_type, labeled_only):
    # type: (str, bool) -> ClassificationDataGenerator
    raise NotImplementedError('get_classification_data_generator has not been implemented')


def main():

    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Training function for material segmentation')
    ap.add_argument('-m', '--model', required=True, type=str, help='Name of the neural network model to use')
    ap.add_argument('-t', '--mtype', required=True, type=str, choices=['segmentation', 'classification'], help='Type of the model')
    ap.add_argument('-e', '--etype', required=True, type=str, choices=['training', 'validation', 'test'], help='Type of the evaluation')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to trainer configuration JSON file')
    ap.add_argument('-w', '--weights', required=True, type=str, help='Path to weights directory or weights file')
    ap.add_argument('--labeledonly', required=False, type=bool, default=False, help='True if the model trained using only labeled data')
    args = vars(ap.parse_args())

    model_name = args['model']
    model_type = args['mtype']
    evaluation_type = args['etype']
    config_file_path = args['config']
    weights_path = args['weights']
    labeled_only = args['labeledonly']

    # Read the configuration file
    global CONFIG
    print 'Loading the configuration from file: {}'.format(config_file_path)
    CONFIG = read_config_json(config_file_path)

    # Build the model
    model = build_model(model_name=model_name, weights_path=weights_path, model_type=model_type)

    # Get the data generator
    if model_type == 'segmentation':
        data_generator = get_segmentation_data_generator(evaluation_type=evaluation_type, labeled_only=labeled_only)
    elif model_type == 'classification':
        data_generator = get_classification_data_generator(evaluation_type=evaluation_type, labeled_only=labeled_only)
    else:
        raise ValueError('Unknown evaluation type: {}'.format(evaluation_type))

    print 'Starting {} set evaluation'.format(evaluation_type)
    stime = time.time()

    results = model.evaluate_generator(
        generator=data_generator,
        steps=data_generator.num_steps_per_epoch,
        workers=dataset_utils.get_number_of_parallel_jobs(),
        trainer=None)

    print 'Evaluation finished in: {} seconds'.format(time.time()-stime)

    for i in range(0, len(results)):
        metric = model.metrics_names[i]
        result = results[i]
        print '{}: {}'.format(metric, result)

    print 'Done'

if __name__ == '__main__':
    main()
