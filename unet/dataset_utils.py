import os
import random
import multiprocessing
import threading

import numpy as np

from PIL import Image
from joblib import Parallel, delayed
from keras.preprocessing.image import load_img, img_to_array


##############################################
# UTILITY CLASSES
##############################################

class MaterialClassInformation:

	def __init__(
		self,
		material_id,
		substance_ids,
		substance_names,
		color_values):
		self.id = material_id

		self.substance_ids = substance_ids
		self.substance_names = substance_names
		self.color_values = color_values	

		self.name = substance_names[0]


##############################################
# UTILITY FUNCTIONS
##############################################

'''
Returns all the files (filenames) found in the path.
Does not include subdirectories.
'''
def get_files(path, ignore_hidden_files=True):
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

	# Filter hidden files such as .DS_Store
	if ignore_hidden_files:
		files = [f for f in files if not f.startswith('.')]
	return files

'''
Normalizes the color channels from the given image to zero-centered
range [-1, 1] from the original [0, 255] range. If the per channels
mean is provided it is subtracted from the image after zero-centering.
Furthermore if the per channel standard deviation is given it is
used to 
'''
def normalize_image_channels(img_array, per_channel_mean=None, per_channel_stddev=None):
	img_array -= 128
	img_array /= 128

	if (per_channel_mean != None):
		# Subtract the per-channel-mean from the batch
		# to "center" the data.
		img_array -= per_channel_mean

	if (per_channel_stddev != None):
		# Additionally, you ideally would like to divide by the sttdev of
		# that feature or pixel as well if you want to normalize each feature
		# value to a z-score.
		img_array /= per_channel_stddev

	# Sanity check for the image values, we shouldn't have any NaN or inf values
	if (np.any(np.isnan(img_array))):
		raise ValueError('NaN values found in image after normalization')

	if (np.any(np.isinf(img_array))):
		raise ValueError('Inf values found in image after normalization')

	return img_array


'''
Calculates the per-channel mean from all the images in the given
path and returns it as a 3 dimensional numpy array.

Parameters to the function:

path - the path to the image files
files - the files to be used in the calculation
'''
def calculate_per_channel_mean(path, files):
	# Continue from saved data if there is
	px_tot = 0.0
	color_tot = np.array([0.0, 0.0, 0.0])

	for idx in range(0, len(files)):
		f = files[idx]

		if idx%10 == 0 and idx != 0:
			print 'Processed {} images: px_tot: {}, color_tot: {}'.format(idx, px_tot, color_tot)
			print 'Current per-channel mean: {}'.format(color_tot/px_tot)

		# Load the image as numpy array
		img = load_img(os.path.join(path, f))
		img_array = img_to_array(img)

		# Normalize colors to zero-centered range [-1, 1]
		img_array = normalize_image_channels(img_array)
		
		# Accumulate the number of total pixels
		px_tot += img_array.shape[0] * img_array.shape[1]

		# Accumulate the sums of the different color channels
		color_tot[0] += np.sum(img_array[:,:,0])
		color_tot[1] += np.sum(img_array[:,:,1])
		color_tot[2] += np.sum(img_array[:,:,2])

	# Calculate the final value
	per_channel_mean = color_tot / px_tot
	print 'Per-channel mean calculation complete: {}'.format(per_channel_mean)

	return per_channel_mean


'''
Calculates the per-channel-stddev
'''
def calculate_per_channel_stddev(path, files, per_channel_mean):
	# Calculate variance
	px_tot = 0.0
	var_tot = np.array([0.0, 0.0, 0.0])

	for idx in range(0, len(files)):
		f = files[idx]

		if idx%10 == 0 and idx != 0:
			print 'Processed {} images: px_tot: {}, var_tot: {}\n'.format(idx, px_tot, var_tot)
			print 'Current per channel variance: {}\n'.format(var_tot/px_tot)

		# Load the image as numpy array
		img = load_img(os.path.join(path, f))
		img_array = img_to_array(img)

		# Normalize colors to zero-centered range [-1, 1]
		img_array = normalize_image_channels(img_array)
		
		# Accumulate the number of total pixels
		px_tot += img_array.shape[0] * img_array.shape[1]

		# Var: SUM_0..N {(val-mean)^2} / N
		var_tot[0] += np.sum(np.square(img_array[:,:,0] - per_channel_mean[0]))
		var_tot[1] += np.sum(np.square(img_array[:,:,1] - per_channel_mean[1]))
		var_tot[2] += np.sum(np.square(img_array[:,:,2] - per_channel_mean[2]))

	# Calculate final variance value
	per_channel_var = var_tot/px_tot
	print 'Final per-channel variance: {}'.format(per_channel_var)

	# Calculate the stddev
	per_channel_stddev = np.sqrt(per_channel_var)
	print 'Per-channel stddev calculation complete: {}'.format(per_channel_stddev)

	return per_channel_stddev


def calculate_mask_class_frequencies(mask_file_path, material_class_information):
	img_array = img_to_array(load_img(mask_file_path))
	expanded_mask = expand_mask(img_array, material_class_information)
	class_pixels = np.sum(expanded_mask, axis=(0,1))

	# Select all classes which appear in the picture i.e.
	# have a value over zero
	num_pixels = img_array.shape[0]*img_array.shape[1]
	img_pixels = (class_pixels > 0.0) * num_pixels

	return (class_pixels, img_pixels)


'''
Calculates the median frequency balancing weights
'''
def calculate_median_frequency_balancing_weights(path, files, material_class_information):
	num_cores = multiprocessing.cpu_count()
	n_jobs = min(32, num_cores)

	data = Parallel(n_jobs=n_jobs, backend='threading')(
		delayed(calculate_mask_class_frequencies)(
			os.path.join(path, f),
			material_class_information) for f in files)
	data = zip(*data)

	class_pixels = data[0]
	img_pixels = data[1]
	class_pixels = np.sum(class_pixels, axis=0)
	img_pixels = np.sum(img_pixels, axis=0)

	# freq(c) is the number of pixels of class c divided
	# by the total number of pixels in images where c is present.
	# Median freq is the median of these frequencies.
	class_frequencies = class_pixels/img_pixels
	median_frequency = np.median(class_frequencies)
	median_frequency_weights = median_frequency/class_frequencies

	return median_frequency_weights


def load_material_class_information(material_labels_file_path):
	materials = []

	with open(material_labels_file_path, 'r') as f:
		content = f.readlines()
	
	# Remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]

	# First line is header so start from 1
	for i in range(1, len(content)):
		
		# Each line is of form
		#
		# substance_id,substance_name,red_color;substance_id,substance_name,red_color
		# The semicolon ';' is used to combine multiple material colors under a
		# single category. In case materials are combined under one category
		subcategories = content[i].split(';')		
		material_params = [f.split(',') for f in subcategories]
		material_params = zip(*material_params)

		substance_ids = [int(x) for x in material_params[0]]
		substance_names = [x for x in material_params[1]]
		color_values = [int(x) for x in material_params[2]]

		# The id is the index of the material in the file, this index will determine
		# the dimension index in the mask image for this material class
		materials.append(MaterialClassInformation(i-1, tuple(substance_ids), tuple(substance_names), tuple(color_values)))

	return materials


'''
The material information in the mask is encoded into the red color channel.
Parameters to the function:

mask - a numpy array of the segmentation mask image WxHx3
material_class_information - an array which has the relevent MaterialClassInformation objects

The functions returns an object of size:

MASK_HEIGHT x MASK_WIDTH x NUM_MATERIAL CLASSES
'''
def expand_mask(mask, material_class_information, verbose=False):
	num_material_classes = len(material_class_information)
	expanded_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], num_material_classes), dtype='float32')
	found_materials = [] if verbose else None

	# Go through each material class
	for material_class in material_class_information:

		# Initialize a color mask with all false
		class_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype='bool')

		# Go through each possible color for that class and create a mask
		# of the pixels that contain a color of the possible values.
		# Note: many colors are possible because some classes maybe collapsed
		# together to form a single class
		for color in material_class.color_values:
			# The substance/material category information is in the red
			# color channel in the opensurfaces dataset
			class_mask |= mask[:,:,0] == color

        # Set the activations of all the pixels that match the color mask to 1
        # on the dimension that matches the material class id
		if (np.any(class_mask)):
			if (found_materials != None):
				found_materials.append(material_class.substance_ids)
			expanded_mask[:,:,material_class.id][class_mask] = 1.0

	if (verbose):
		print 'Found {} materials with the following substance ids: {}\n'.format(len(found_materials), found_materials)

	return expanded_mask



'''
Splits the whole dataset randomly into three different groups: training,
validation and test, according to the split provided as the parameter.

Returns three lists of photo - mask pairs:

0 training
1 validation
2 test
'''
def split_dataset(
	photo_files,
	mask_files,
	split):

	if (len(photo_files) != len(mask_files)):
		raise ValueError('Unmatching photo - mask file list sizes: photos: {}, masks: {}'.format(len(photo_files), len(mask_files)))

	if (sum(split) != 1.0):
		raise ValueError('The given dataset split does not sum to 1: {}'.format(sum(split)))

	# Sort the files by name so we have matching photo - mask files
	photo_files.sort()
	mask_files.sort()

	# Zip the lists to create a list of matching photo - mask file tuples
	photo_mask_files = zip(photo_files, mask_files)

	# Shuffle the list of files
	random.shuffle(photo_mask_files)

	# Divide the dataset to three different parts: training, validation and test
	# according to the given split: 0=training, 1=validation, 2=test
	dataset_size = len(photo_mask_files)
	training_set_size = int(round(split[0] * dataset_size))
	validation_set_size = int(round(split[1] * dataset_size))
	test_set_size = int(round(split[2] * dataset_size))
	
	# If the sizes don't match exactly add/subtract the different
	# from the training set
	if (training_set_size + validation_set_size + test_set_size != dataset_size):
		diff = dataset_size - (training_set_size + validation_set_size + test_set_size)
		training_set_size += diff

	if (training_set_size + validation_set_size + test_set_size != dataset_size):
		raise ValueError('The split set sizes do not sum to total dataset size: {} + {} + {} = {} != {}'.format(training_set_size, validation_set_size, test_set_size, training_set_size + validation_set_size + test_set_size, dataset_size))

	training_set = photo_mask_files[0:training_set_size]
	validation_set = photo_mask_files[training_set_size:training_set_size+validation_set_size]
	test_set = photo_mask_files[training_set_size+validation_set_size:]

	return training_set, validation_set, test_set
