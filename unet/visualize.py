import sys
import json
import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters

import unet
import dataset_utils

CONFIG = None

##############################################
# UTILITIES
##############################################

def read_config_json(path):
	with open(path) as f:
		data = f.read()
		return json.loads(data)

def get_config_value(key):
	global CONFIG
	return CONFIG[key] if key in CONFIG else None

def set_config_value(key, value):
	global CONFIG
	CONFIG[key] = value

def get_layer_idx(model, layer_name):
	return [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

##############################################
# MAIN
##############################################

if __name__ == '__main__':

	if (len(sys.argv) < 4):
		print 'Invalid number of arguments, usage: ./python {} <config_file> <weights_file> <layer_name>'.format(sys.argv[0])
		sys.exit(0)

	config_file = sys.argv[1]
	weights_file = sys.argv[2]
	layer_name = sys.argv[3]

	print 'Loading the configuration from file: {}'.format(config_file)
	CONFIG = read_config_json(config_file)

	# Load the material class information
	material_class_information_path = get_config_value('path_to_material_class_file')
	print 'Loading the material class information from file: {}'.format(material_class_information_path)
	material_class_information = dataset_utils.load_material_class_information(material_class_information_path)
	print 'Loaded {} material classes'.format(len(material_class_information))

	print 'Loading the model'
	model = unet.get_unet((256, 256, get_config_value('num_channels')), len(material_class_information))

	print 'Loading weights from file: {}'.format(weights_file)
	model.load_weights(weights_file)

	print 'Searching for layer: {}'.format(layer_name)
	layer_idx = get_layer_idx(model, layer_name)
	layer = model.get_layer(name=layer_name)
	print 'Found layer at index: {}'.format(layer_idx)

	# Visualize all filters in this layer.
	filters = np.arange(get_num_filters(layer))

	print 'Visualizing {} filters'.format(len(filters))

	# Generate input image for each filter.
	# Here `text` field is used to overlay `filter_value` on top of the image.
	vis_images = []

	for idx in filters:
		img = visualize_activation(model, layer_idx, filter_indices=idx)
		img = utils.draw_text(img, str(idx))
		vis_images.append(img)

	# Generate stitched image palette with 8 cols.
	stitched = utils.stitch_images(vis_images, cols=8)
	plt.axis('off')
	plt.imshow(stitched)
	plt.title(layer_name)
	plt.show()
