# coding=utf-8

import json
import sys
import numpy as np

from quiver_engine import server

from utils import training_utils
from utils.training_utils import get_config_value

from utils import dataset_utils
from models import model_utils


##############################################
# MAIN
##############################################

if __name__ == '__main__':

    # Read the configuration file and make it global
    if len(sys.argv) < 2:
        print 'Invalid number of parameters, usage: python {} <config.json>'.format(sys.argv[0])
        sys.exit(0)

    training_utils.CONFIG = training_utils.read_config_json(sys.argv[1])
    training_utils.LOG_FILE_PATH = get_config_value('log_file_path')
    print 'Configuration file read successfully'

    # Load material class information
    print 'Loading material class information'
    material_class_information = dataset_utils.load_material_class_information(
        get_config_value('path_to_material_class_file'))
    num_classes = len(material_class_information)
    print 'Loaded {} material classes successfully'.format(num_classes)

    # Create the model
    model_name = get_config_value('model')
    num_channels = get_config_value('num_channels')
    input_shape = (None, None, num_channels)

    print 'Creating model {} instance with input shape: {}, num classes: {}'\
        .format(model_name, input_shape, num_classes)

    model = model_utils.get_model(model_name, input_shape, num_classes)
    model.summary()

    # Load the weights
    initial_epoch = training_utils.load_latest_weights(
        get_config_value('keras_model_checkpoint_file_path'),
        model)

    print 'Loaded weights from epoch {}'.format(initial_epoch-1)

    print 'Launching quiver server'
    server.launch(model, input_folder='../photos/test-photos')
