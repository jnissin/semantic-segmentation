import numpy as np
import random
import os
import math
import datetime
import time
import multiprocessing
import threading
import json
import sys

import unet
import dataset_utils

from joblib import Parallel, delayed

from keras import backend as K
from keras import metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD, Adam

from PIL import Image
from PIL import ImageFile

##############################################
# GLOBALS
##############################################

CONFIG = None
LOG_FILE = None
LOG_FILE_PATH = None

##############################################
# UTILITIES
##############################################

def log(s, log_to_stdout=True):
	global LOG_FILE
	global LOG_FILE_PATH

	# Create and open the log file
	if not LOG_FILE:
		if LOG_FILE_PATH:
			if not os.path.exists(os.path.dirname(LOG_FILE_PATH)):
				os.makedirs(os.path.dirname(LOG_FILE_PATH))

			LOG_FILE = open(LOG_FILE_PATH, 'w')
		else:
			raise ValueError('The log file path is None, cannot log')

	# Log to file - make sure there is a newline
	if not s.endswith('\n'):
		LOG_FILE.write(s + "\n")
	else:
		LOG_FILE.write(s)

	# Log to stdout - no newline needed
	if log_to_stdout:
		print s.strip()


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


def get_latest_weights_file_path(weights_folder_path):
	weight_files = dataset_utils.get_files(weights_folder_path)

	if len(weight_files) > 0:
		weight_files.sort()
		weight_file = weight_files[-1]
		return os.path.join(weights_folder_path, weight_file)

	return None

##############################################
# GENERATOR FUNCTIONS
##############################################

def get_photo_mask_crop_pair(
	photos_files_folder_path,
	mask_files_folder_path,
	photo_mask_pair,
	per_channel_mean,
	per_channel_stddev,
	material_class_information,
	crop_size):

	image = load_img(os.path.join(photos_files_folder_path, photo_mask_pair[0]))
	mask = load_img(os.path.join(mask_files_folder_path, photo_mask_pair[1]))

	# Resize the image to match the mask size if necessary, since
	# the original photos are sometimes huge
	if (image.size != mask.size):
		orig_size = image.size
		image = image.resize(mask.size, Image.ANTIALIAS)

	if (image.size != mask.size):
		raise ValueError('Non-matching image and mask dimensions after resize: {} vs {}'
			.format(image.size, mask.size))

	# If a crop size is given:
	# Take a random crop of both the image and the mask
	if crop_size != None:

		if (crop_size[0]%2 != 0 or crop_size[1]%2 != 0):
			raise ValueError('The crop size is not a multiple of two - this will cause problems during upsampling')

		try:
			# Re-attempt crops if the crops end up getting only black pixels
			attempts = 5

			for i in range(0, attempts):
				x1 = np.random.randint(0, image.size[0]-crop_size[0])
				y1 = np.random.randint(0, image.size[1]-crop_size[1])
				x2 = x1 + crop_size[0]
				y2 = y1 + crop_size[1]

				mask_crop = img_to_array(mask.crop((x1, y1, x2, y2)))
				
				# If the mask crop is only background (all R channel is zero) - try another crop
				if (np.max(mask_crop[:,:,0]) == 0 and i < attempts-1):
					continue
				
				mask = mask_crop
				image = img_to_array(image.crop((x1, y1, x2, y2)))
				break

		except IOError:
			log('ERROR: Could not load image or mask from pair: {}, {}'.format(photo_mask_pair[0], photo_mask_pair[1]))
			raise IOError('Could not load image or mask from pair: {}, {}'.format(photo_mask_pair[0], photo_mask_pair[1]))

	# If a crop size is not given:
	# Make sure the image has height and width that are divisible by two
	# otherwise the upsampling is going to screw up the dimensions
	else:
		image = img_to_array(image)
		mask = img_to_array(mask)

		if (image.shape[0]%2 != 0):
			image = image[:-1]
			mask = mask[:-1]

		if (image.shape[1]%2 != 0):
			image = image[:,:-1]
			mask = mask[:,:-1]

	# Normalize the color channels of the original images
	# to zero centered range [-1,1]
	image = dataset_utils.normalize_image_channels(image, per_channel_mean, per_channel_stddev)

	# Expand the mask image to accommodate different classes
	# H x W x NUM_CLASSES
	mask = dataset_utils.expand_mask(mask, material_class_information)

	return image, mask


'''
Takes an iterator/generator and makes it thread-safe by
serializing call to the `next` method of given iterator/generator.
'''
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


'''
A decorator that takes a generator function and makes it thread-safe.
'''
def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


'''
Generates inifinite batches of data for training/validation from the files
provided to the generator. Use this to get generators for validation and
training data sets.
'''
@threadsafe_generator
def get_generator(
	photos_files_folder_path,
	mask_files_folder_path,
	photo_mask_files,
	per_channel_mean,
	per_channel_stddev,
	material_class_information,
	crop_size,
	batch_size):

	# Calculate the number of batches that we can create from this data
	num_batches = len(photo_mask_files) // batch_size

	# Calculate number of cores available
	num_cores = multiprocessing.cpu_count()
	n_jobs = min(32, num_cores)

	while True:
		# Shuffle the photo - mask pairs
		random.shuffle(photo_mask_files)

		for i in range(0, num_batches):
			# The files for this batch
			batch_files = photo_mask_files[i*batch_size:(i+1)*batch_size]

			# Parallel processing of the files in this batch
			data = Parallel(n_jobs=n_jobs, backend='threading')(
				delayed(get_photo_mask_crop_pair)(
					photos_files_folder_path,
					mask_files_folder_path,
					pair,
					per_channel_mean,
					per_channel_stddev,
					material_class_information,
					crop_size) for pair in batch_files)

			# Note: all the examples in the batch have to have the same dimensions
			X, Y = zip(*data)
			X, Y = np.array(X), np.array(Y)

			yield X, Y


##############################################
# MAIN
##############################################

if __name__ == '__main__':

	# Read the configuration file and make it global
	if len(sys.argv) < 2:
		print 'Invalid number of parameters, usage: python {} <config.json>'.format(sys.argv[0])
		sys.exit(0)

	# Without this some truncated images can throw errors
	ImageFile.LOAD_TRUNCATED_IMAGES = True

	CONFIG = read_config_json(sys.argv[1])
	LOG_FILE_PATH = get_config_value('log_file_path')
	print 'Configuration file read successfully'

	# Setup the global LOG_FILE_PATH to enable logging
	log('\n\n############################################################\n')
	log('Starting a new session at local time {}\n'.format(datetime.datetime.now()))

	# Seed the random in order to be able to reproduce the results
	# Note: both random and np.random
	log('Starting program with random seed: {}'.format(get_config_value('random_seed')))
	random.seed(get_config_value('random_seed'))
	np.random.seed(get_config_value('random_seed'))

	# Set image data format
	log('Setting Keras image data format to: {}'.format(get_config_value('image_data_format')))
	K.set_image_data_format(get_config_value('image_data_format'))

	log('Loading material class information')
	material_class_information = dataset_utils.load_material_class_information(get_config_value('path_to_material_class_file'))
	num_classes = len(material_class_information)
	log('Loaded {} material classes successfully'.format(num_classes))

	# Read the data
	log('Reading photo files from: {}'.format(get_config_value('path_to_photos')))
	photo_files = dataset_utils.get_files(get_config_value('path_to_photos'))
	log('Found {} photo files'.format(len(photo_files)))

	log('Reading mask files from: {}'.format(get_config_value('path_to_masks')))
	mask_files = dataset_utils.get_files(get_config_value('path_to_masks'))
	log('Found {} mask files'.format(len(mask_files)))

	if (len(photo_files) != len(mask_files)):
		raise ValueError('Unmatching photo - mask file list sizes: photos: {}, masks: {}'.format(len(photo_files), len(mask_files)))

	# Generate random splits of the data for training, validation and test
	log('Splitting data to training, validation and test sets of sizes (%) of the whole dataset of size {}: {}'.format(len(photo_files), get_config_value('dataset_splits')))
	training_set, validation_set, test_set = dataset_utils.split_dataset(
		photo_files,
		mask_files,
		get_config_value('dataset_splits'))

	log('Dataset split complete')
	log('Training set size: {}'.format(len(training_set)))
	log('Validation set size: {}'.format(len(validation_set)))
	log('Test set size: {}'.format(len(test_set)))

	log('Saving the dataset splits to log file\n')
	log('training_set: {}\n'.format(training_set), False)
	log('validation_set: {}\n'.format(validation_set), False)
	log('test_set: {}\n'.format(test_set), False)

	# Calculate the per-channel mean of the training set photos
	if get_config_value('per_channel_mean') == None:
		log('Existing per-channel mean was not found')
		log('Calculating per-channel mean from {} training set photos'.format(len(training_set)))
		
		training_set_photos = [sample[0] for sample in training_set]
		pcm = dataset_utils.calculate_per_channel_mean(get_config_value('path_to_photos'), training_set_photos)
		set_config_value('per_channel_mean', pcm)
		
		log('Per-channel mean calculation complete: {}'.format(pcm))
	else:
		log('Using existing per-channel mean: {}'.format(get_config_value('per_channel_mean')))

	# Calculate the per-channel standard deviation of the training set photos
	if get_config_value('per_channel_stddev') == None:
		log('Existing per-channel stddev was not found')
		log('Calculating per-channel stddev from {} training set photos with per-channel mean: {}'.format(len(training_set), get_config_value('per_channel_mean')))
		
		training_set_photos = [sample[0] for sample in training_set]
		pcs = dataset_utils.calculate_per_channel_stddev(get_config_value('path_to_photos'), training_set_photos, get_config_value('per_channel_mean'))
		set_config_value('per_channel_stddev', pcs)

		log('Per-channel stddev calculation complete: {}'.format(pcs))
	else:
		log('Using existing per-channel stddev: {}'.format(get_config_value('per_channel_stddev')))

	# Calculate the median frequency balancing weights
	median_frequency_balancing_weights = get_config_value('median_frequency_balancing_weights')

	if median_frequency_balancing_weights == None or len(median_frequency_balancing_weights) != len(material_class_information):
		log('Median frequency balancing weights were not found or did not match the number of material classes')
		log('Calculating median frequency balancing weights for the training set')
		training_set_masks = [sample[1] for sample in training_set]
		median_frequency_balancing_weights = dataset_utils.calculate_median_frequency_balancing_weights(get_config_value('path_to_masks'), training_set_masks, material_class_information)
		log('Median frequency balancing weights calculated: {}'.format(median_frequency_balancing_weights))
	else:
		log('Using existing median frequency balancing weights: {}'.format(median_frequency_balancing_weights))
		median_frequency_balancing_weights = np.array(median_frequency_balancing_weights)

	# Create optimizer
	optimizer = None

	if get_config_value('optimizer') == 'adam':
		lr = get_config_value('learning_rate')
		decay = get_config_value('decay')
		optimizer = Adam(lr=lr, decay=decay)
		log('Using Adam optimizer with learning rate: {}, decay: {}'.format(lr, decay))
	elif get_config_value('optimizer') == 'sgd':
		lr = get_config_value('learning_rate')
		decay = get_config_value('decay')
		momentum = get_config_value('momentum')
		optimizer = SGD(lr=lr, momentum=momentum, decay=decay)
		log('Using SGD optimizer with learning rate: {}, momentum: {}, decay: {}'.format(lr, momentum, decay))
	else:
		log('Unknown optimizer: {} exiting'.format(get_config_value('optimizer')))
		sys.exit(0)

	log('Creating model instance with {} input channels and {} classes'.format(get_config_value('num_channels'), num_classes))
	model = unet.get_unet((None, None, get_config_value('num_channels')), num_classes)

	log('Compiling model')
	model.compile(
    	optimizer=optimizer,
        loss=unet.pixelwise_crossentropy,#unet.weighted_pixelwise_crossentropy(median_frequency_balancing_weights),
        metrics=['accuracy', unet.mean_iou(num_classes), unet.mean_per_class_accuracy(num_classes)])
    
	model.summary()

	# Look for a crop size
	crop_size = None
	
	if (get_config_value('crop_width') == None or
		get_config_value('crop_height') == None):
		crop_size = None
	else:
		crop_size = (get_config_value('crop_width'), get_config_value('crop_height'))

	log('Creating training data generator')
	train_generator = get_generator(
		get_config_value('path_to_photos'),
		get_config_value('path_to_masks'),
		training_set,
		get_config_value('per_channel_mean'),
		get_config_value('per_channel_stddev'),
		material_class_information,
		crop_size,
		get_config_value('batch_size'))

	log('Creating validation data generator')
	validation_generator = get_generator(
		get_config_value('path_to_photos'),
		get_config_value('path_to_masks'),
		validation_set,
		get_config_value('per_channel_mean'),
		get_config_value('per_channel_stddev'),
		material_class_information,
		crop_size,
		get_config_value('batch_size'))

	num_epochs = get_config_value('num_epochs')
	batch_size = get_config_value('batch_size')
	training_set_size = len(training_set)
	validation_set_size = len(validation_set)
	steps_per_epoch = training_set_size // batch_size
	validation_steps = validation_set_size // batch_size

	log('Starting training for {} epochs with batch size: {}, crop_size: {}, steps per epoch: {}, validation steps: {}'
		.format(num_epochs, batch_size, crop_size, steps_per_epoch, validation_steps))

	# Model checkpoint callback to save model on every epoch
	model_checkpoint_callback = ModelCheckpoint(
		filepath=get_config_value('keras_model_checkpoint_file_path'),
		monitor='val_loss',
		verbose=1,
		save_best_only=False,
		save_weights_only=False,
		mode='auto',
		period=1)

	# Tensorboard checkpoint callback to save on every epoch
	tensorboard_checkpoint_callback = TensorBoard(
		log_dir=get_config_value('keras_tensorboard_log_path'),
		histogram_freq=1,
		write_graph=True,
		write_images=True,
		embeddings_freq=0,
		embeddings_layer_names=None,
		embeddings_metadata=None)

	# CSV logger for streaming epoch results
	csv_logger_callback = CSVLogger(
		get_config_value('keras_csv_log_file_path'),
		separator=',',
		append=False)

	# Load existing weights to cotinue training
	initial_epoch = 0

	if (get_config_value('continue_from_last_checkpoint')):
		# Try to find weights from the checkpoint path
		weights_folder = os.path.dirname(get_config_value('keras_model_checkpoint_file_path'))
		log('Searching for existing weights from checkpoint path: {}'.format(weights_folder))
		weight_file_path = get_latest_weights_file_path(weights_folder)
		weight_file = weight_file_path.split('/')[-1]

		if weight_file:
			log('Loading weights from file: {}'.format(weight_file_path))
			model.load_weights(weight_file_path)

			# Parse the epoch number: <epoch>-<vloss>
			epoch_val_loss = weight_file.split('.')[1]
			initial_epoch = int(epoch_val_loss.split('-')[0]) + 1
			log('Continuing training from epoch: {}'.format(initial_epoch))

		else:
			log('No existing weights were found')

	model.fit_generator(train_generator,
		steps_per_epoch=steps_per_epoch,
		epochs=num_epochs,
		initial_epoch=initial_epoch,
		validation_data=validation_generator,
		validation_steps=validation_steps,
		verbose=1,
		callbacks=[model_checkpoint_callback, tensorboard_checkpoint_callback, csv_logger_callback])

	log('The session ended at local time {}\n'.format(datetime.datetime.now()))

	# Close the log file
	if LOG_FILE:
		LOG_FILE.close()