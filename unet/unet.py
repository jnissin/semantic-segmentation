import numpy as np
import random
import os
import math
import datetime
import time
import multiprocessing
import threading

from joblib import Parallel, delayed

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing.image import load_img, img_to_array

from PIL import Image

##############################################
# GLOBALS
##############################################

PATH_TO_PHOTOS = '/Volumes/Omenakori/opensurfaces/photos-resized/'
PATH_TO_MASKS = '/Volumes/Omenakori/opensurfaces/photos-labels/'
SAVED_WEIGHTS_PATH = 'unet/weights/'
SAVED_CHECKPOINTS_PATH = 'unet/checkpoints/'

KERAS_TENSORBOARD_LOG_PATH = 'unet/logs/tensorboard/'
KERAS_MODEL_CHECKPOINT_FILE_PATH = 'unet/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
KERAS_CSV_LOG_FILE_PATH = 'unet/logs/unet-per-epoch-data.csv'

LOG_FILE_PATH = 'unet/logs/log.txt'
LOG_FILE = None
LOG_TO_STDOUT = True

IMAGE_DATA_FORMAT = 'channels_last' 							# Image data format for Keras. Tensorflow uses channels last, Theano channels first

NUM_EPOCHS = 200 												# How many epochs are we training for?
BATCH_SIZE = 1 													# Batch size for training - power of 2s are faster
LEARNING_RATE = 10e-4											# Learning rate, also known as step size
TILE_WIDTH = 512 												# Width of a single input image tile when training the network, original U-net: 572px
TILE_HEIGHT = 512												# Height of a single input image tile when training the network, original U-net: 572px

DATASET_SPLITS = [0.8, 0.05, 0.15] 								# Sizes (%) of training, validation and test set
NUM_CHANNELS = 3 												# 3 for RGB, 4 for RGB+NIR

RANDOM_SEED = 14874
PER_CHANNEL_MEAN = (-0.0748563, -0.20287086, -0.3310074)		# In zero-centered range [-1,1]
PER_CHANNEL_MEAN_255 = (236, 203, 171)							# In RGB range [0, 255]
PER_CHANNEL_VARIANCE = (0.28665781, 0.27211733, 0.27841234)		# In zero-centered range [-1,1]
PER_CHANNEL_STDDEV = (0.53540435, 0.52164866, 0.52764793)		# In zero-centered range [-1,1]

IMAGE_MASK_PAIR_SAVE_PATH = 'unet/logs/img_mask_crop_pairs/'
NUM_IMAGE_MASK_PAIRS_TO_SAVE = 25000
SAVED_IMAGE_MASK_PAIR_IDX = 0

##############################################
# UTILITIES
##############################################

def log(s):
	global LOG_FILE
	global LOG_FILE_PATH

	# Create and open the log file
	if not LOG_FILE:
		if LOG_FILE_PATH:
			if not os.path.exists(os.path.dirname(LOG_FILE_PATH)):
				os.makedirs(os.path.dirname(LOG_FILE_PATH))

			LOG_FILE = open(LOG_FILE_PATH, 'a+')
		else:
			raise ValueError('The log file path is None, cannot log')

	# Log to file - make sure there is a newline
	if not s.endswith('\n'):
		LOG_FILE.write(s + "\n")
	else:
		LOG_FILE.write(s)

	# Log to stdout - no newline needed
	if LOG_TO_STDOUT:
		print s.strip()





'''
Returns all the files (filenames) found in the path.
Does not include subdirectories.
'''
def get_files(path, ignore_hidden_files=True):
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	if ignore_hidden_files:
		files = [f for f in files if not f.startswith('.')]
	return files


'''
Normalizes the color channels from the given image to zero-centered
range [-1, 1] from the original [0, 255] range.
'''
def normalize_image_channels(img_array):
	img_array -= 128
	img_array /= 128
	return img_array


'''
Calculates the per-channel mean from all the images in the given
path and returns it as a 3 dimensional numpy array.

Parameters to the function:

path - the path to the image files
files - the files to be used in the calculation
log_file_path - log file path if you want logging
'''
def calculate_per_channel_mean(path, files):
	# Continue from saved data if there is
	px_tot = 0.0
	color_tot = np.array([0.0, 0.0, 0.0])

	for idx in range(0, len(files)):
		f = files[idx]

		if idx%10 == 0 and idx != 0:
			log('Processed {} images: px_tot: {}, color_tot: {}\n'.format(idx, px_tot, color_tot))
			log('Current per-channel mean: {}\n'.format(color_tot/px_tot))

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

	log('Per-channel mean calculation complete: {}\n'.format(per_channel_mean))
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
			log('Processed {} images: px_tot: {}, var_tot: {}\n'.format(idx, px_tot, var_tot))
			log('Current per channel variance: {}\n'.format(var_tot/px_tot))

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

	log('Final per-channel variance: {}\n'.format(per_channel_var))

	# Calculate the stddev
	per_channel_stddev = np.sqrt(per_channel_var)

	log('Per-channel stddev calculation complete: {}\n'.format(per_channel_stddev))

	return per_channel_stddev
	

##############################################
# DATASET FUNCTIONS
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

MASK_WIDTH x MASK_HEIGHT x NUM_MATERIAL CLASSES
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
			if (found_materials):
				found_materials.append(material_class.substance_ids)
			expanded_mask[:,:,material_class.id][class_mask] = 1.0

	if (verbose):
		log('Found {} materials with the following substance ids: {}\n'.format(len(found_materials), found_materials))

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

	log('Saving the dataset split\n')
	log('random seed: {}\n'.format(RANDOM_SEED))
	log('training_set: {}\n'.format(training_set))
	log('validation_set: {}\n'.format(validation_set))
	log('test_set: {}\n'.format(test_set))

	return training_set, validation_set, test_set


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
		start_time = time.time()
		orig_size = image.size
		image = image.resize(mask.size, Image.ANTIALIAS)
		print 'Resize from {} to {} took: {}'.format(orig_size, image.size, time.time()-start_time)

	if (image.size != mask.size):
		raise ValueError('Non-matching image and mask dimensions after resize: {} vs {}'
			.format(image.size, mask.size))

	# Take a random crop of both the image and the mask
	x1 = random.randrange(0, image.size[0]-crop_size[0])
	y1 = random.randrange(0, image.size[1]-crop_size[1])
	x2 = x1 + crop_size[0]
	y2 = y1 + crop_size[1]

	image = image.crop((x1, y1, x2, y2))
	mask = mask.crop((x1, y1, x2, y2))

	# Logging for image - mask crop pairs
	global NUM_IMAGE_MASK_PAIRS_TO_SAVE
	global SAVED_IMAGE_MASK_PAIR_IDX
	global IMAGE_MASK_PAIR_SAVE_PATH

	if (NUM_IMAGE_MASK_PAIRS_TO_SAVE != 0 and IMAGE_MASK_PAIR_SAVE_PATH):
		# Create the path if necessary
		if not os.path.exists(os.path.dirname(IMAGE_MASK_PAIR_SAVE_PATH)):
			os.makedirs(os.path.dirname(IMAGE_MASK_PAIR_SAVE_PATH))

		image.save(os.path.join(IMAGE_MASK_PAIR_SAVE_PATH, '{}_photo.jpg'.format(SAVED_IMAGE_MASK_PAIR_IDX)))
		mask.save(os.path.join(IMAGE_MASK_PAIR_SAVE_PATH, '{}_mask.png'.format(SAVED_IMAGE_MASK_PAIR_IDX)))
		SAVED_IMAGE_MASK_PAIR_IDX = (1 + SAVED_IMAGE_MASK_PAIR_IDX)%NUM_IMAGE_MASK_PAIRS_TO_SAVE

	image = img_to_array(image)
	mask = img_to_array(mask)

	# Normalize the color channels of the original images
	# to zero centered range [-1,1]
	image = normalize_image_channels(image)

	# Subtract the per-channel-mean from the batch
	# to "center" the data.
	image -= per_channel_mean

	# Additionally, you ideally would like to divide by the sttdev of
	# that feature or pixel as well if you want to normalize each feature
	# value to a z-score.
	image /= per_channel_stddev

	# Expand the mask image to accommodate different classes
	# W x H x NUM_CLASSES
	mask = expand_mask(mask, material_class_information)

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

	# Shuffle the photo - mask pairs
	random.shuffle(photo_mask_files)

	# Calculate the number of batches that we can create from this data
	num_batches = len(photo_mask_files) // BATCH_SIZE

	# Calculate number of cores available
	num_cores = multiprocessing.cpu_count()
	n_jobs = min(32, num_cores)

	while True:
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
# UNET
##############################################

'''
Keras softmax doesn't work for N-dimensional tensors. The function
takes in a keras matrix of size WxHxNUM_CLASSES and applies
'depth-wise' softmax to the matrix. In other words the output is a
matrix of size WxHxNUM_CLASSES where for each WxH entry the depth slice
of NUM_CLASSES entries sum to 1.
'''
def depth_softmax(matrix):
	sigmoid = lambda x: 1.0 / (1.0 + K.exp(-x))
	sigmoided_matrix = sigmoid(matrix)
	softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=-1, keepdims=True)
	return softmax_matrix


'''
The functions builds the U-net model presented in the paper:
https://arxiv.org/pdf/1505.04597.pdf
'''
def get_unet(num_channels, num_classes):
    
    '''
    Contracting path
    '''
    # H x W x CHANNELS
    # None, None - means we support variable sized images, however
    # each image within one minibatch during training has to have
    # the same dimensions
    inputs = Input(shape=(None, None, num_channels))
        
    # First convolution layer: 64 output filters
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    # Second convolution layer: 128 output filters
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    # Third convolution layer: 256 output filters
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    # Fourth convolution layer: 512 output filters
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Fifth convolution layer: 1024 output filters
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    '''
    Expansive path
    '''
    # First upsampling layer: 512 output filters
    up6 = UpSampling2D(size=(2,2))(conv5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    concat6 = concatenate([conv6, conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    # Second upsampling layer: 256 output filters
    up7 = UpSampling2D(size=(2,2))(conv6)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    concat7 = concatenate([conv7, conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    # Third upsampling layer: 128 output filters
    up8 = UpSampling2D(size=(2,2))(conv7)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    concat8 = concatenate([conv8, conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    # Fourth upsampling layer: 64 output filters
    up9 = UpSampling2D(size=(2,2))(conv8)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    concat9 = concatenate([conv9, conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    softmax = Lambda(depth_softmax)(conv10)

    model = Model(inputs=inputs, outputs=softmax)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

##############################################
# MAIN
##############################################

if __name__ == '__main__':

	log('\n\n############################################################\n')
	log('Starting a new u-net session at local time {}\n'.format(datetime.datetime.now()))

	# Seed the random in order to be able to reproduce the results
	log('Starting program with random seed: {}'.format(RANDOM_SEED))
	random.seed(RANDOM_SEED)

	# Set image data format
	log('Setting Keras image data format to: {}'.format(IMAGE_DATA_FORMAT))
	K.set_image_data_format(IMAGE_DATA_FORMAT)

	log('Loading material class information')
	material_class_information = load_material_class_information('materials-minc.csv')
	num_classes = len(material_class_information)
	log('Loaded {} material classes successfully'.format(num_classes))

	# Read the data
	log('Reading photo files from: {}'.format(PATH_TO_PHOTOS))
	photo_files = get_files(PATH_TO_PHOTOS)
	log('Found {} photo files'.format(len(photo_files)))

	log('Reading mask files from: {}'.format(PATH_TO_MASKS))
	mask_files = get_files(PATH_TO_MASKS)
	log('Found {} mask files'.format(len(mask_files)))

	if (len(photo_files) != len(mask_files)):
		raise ValueError('Unmatching photo - mask file list sizes: photos: {}, masks: {}'.format(len(photo_files), len(mask_files)))

	# Generate random splits of the data for training, validation and test
	log('Splitting data to training, validation and test sets of sizes (%) of the whole dataset of size {}: {}'.format(len(photo_files), DATASET_SPLITS))
	training_set, validation_set, test_set = split_dataset(
		photo_files,
		mask_files,
		DATASET_SPLITS)

	log('Dataset split complete')
	log('Training set size: {}'.format(len(training_set)))
	log('Validation set size: {}'.format(len(validation_set)))
	log('Test set size: {}'.format(len(test_set)))

	# Calculate the per-channel mean of the training set photos
	if not PER_CHANNEL_MEAN:
		log('Existing per-channel mean was not found')
		log('Calculating per-channel mean from {} training set photos'.format(len(training_set)))
		PER_CHANNEL_MEAN = calculate_per_channel_mean(PATH_TO_PHOTOS, [sample[0] for sample in training_set])
		log('Per-channel mean calculation complete: {}'.format(PER_CHANNEL_MEAN))
	else:
		log('Using existing per-channel mean: {}'.format(PER_CHANNEL_MEAN))

	# Calculate the per-channel standard deviation of the training set photos
	if not PER_CHANNEL_STDDEV:
		log('Existing per-channel stddev was not found')
		log('Calculating per-channel stddev from {} training set photos with per-channel mean: {}'.format(len(training_set), PER_CHANNEL_MEAN))
		PER_CHANNEL_STDDEV = calculate_per_channel_stddev(PATH_TO_PHOTOS, [sample[0] for sample in training_set], PER_CHANNEL_MEAN)
		log('Per-channel stddev calculation complete: {}'.format(PER_CHANNEL_STDDEV))
	else:
		log('Using existing per-channel stddev: {}'.format(PER_CHANNEL_STDDEV))

	log('Creating u-net model instance with {} input channels and {} classes'.format(NUM_CHANNELS, num_classes))
	model = get_unet(NUM_CHANNELS, num_classes)
	model.summary()

	log('Creating training data generator')
	train_generator = get_generator(
		PATH_TO_PHOTOS,
		PATH_TO_MASKS,
		training_set,
		PER_CHANNEL_MEAN,
		PER_CHANNEL_STDDEV,
		material_class_information,
		(TILE_WIDTH, TILE_HEIGHT),
		BATCH_SIZE)

	log('Creating validation data generator')
	validation_generator = get_generator(
		PATH_TO_PHOTOS,
		PATH_TO_MASKS,
		validation_set,
		PER_CHANNEL_MEAN,
		PER_CHANNEL_STDDEV,
		material_class_information,
		(TILE_WIDTH, TILE_HEIGHT),
		BATCH_SIZE)

	training_set_size = len(training_set)
	validation_set_size = len(validation_set)
	steps_per_epoch = training_set_size // BATCH_SIZE
	validation_steps = validation_set_size // BATCH_SIZE

	log('Starting training for {} epochs with batch size: {}, steps per epoch: {}, validation steps: {}'
		.format(NUM_EPOCHS, BATCH_SIZE, steps_per_epoch, validation_steps))

	# Model checkpoint callback to save model on every epoch
	model_checkpoint_callback = ModelCheckpoint(
		filepath=KERAS_MODEL_CHECKPOINT_FILE_PATH,
		monitor='val_loss',
		verbose=1,
		save_best_only=False,
		save_weights_only=False,
		mode='auto',
		period=1)

	# Tensorboard checkpoint callback to save on every epoch
	tensorboard_checkpoint_callback = TensorBoard(
		log_dir=KERAS_TENSORBOARD_LOG_PATH,
		histogram_freq=1,
		write_graph=True,
		write_images=True,
		embeddings_freq=0,
		embeddings_layer_names=None,
		embeddings_metadata=None)

	# CSV logger for streaming epoch results
	csv_logger_callback = CSVLogger(
		KERAS_CSV_LOG_FILE_PATH,
		separator=',',
		append=False)

	model.fit_generator(train_generator,
		steps_per_epoch=steps_per_epoch,
		epochs=NUM_EPOCHS,
		validation_data=validation_generator,
		validation_steps=validation_steps,
		verbose=1,
		callbacks=[model_checkpoint_callback, tensorboard_checkpoint_callback, csv_logger_callback])

	log('U-net session ended at local time {}\n'.format(datetime.datetime.now()))

	# Close the log file
	LOG_FILE.close()