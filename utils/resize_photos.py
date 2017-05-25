'''
This script resizes all the images to match the mask sizes.
'''

import os
import multiprocessing
import time

from PIL import Image
from joblib import Parallel, delayed

PATH_TO_PHOTOS = '/Volumes/Omenakori/opensurfaces/photos/'
PATH_TO_MASKS = '/Volumes/Omenakori/opensurfaces/photos-labels/'
PATH_TO_RESIZED_PHOTOS = '/Volumes/Omenakori/opensurfaces/photos-resized/'

def get_files(path, ignore_hidden_files=True):
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	if ignore_hidden_files:
		files = [f for f in files if not f.startswith('.')]
	return files


def resize(path_to_photos, photo_filename, path_to_masks, mask_filename, path_to_resized_folder):
	if (photo_filename.split('.')[0] != mask_filename.split('.')[0]):
		raise ValueError('Unmatching photo and mask filenames: {} vs {}'.format(photo_filename.split('.')[0], mask_filename.split('.')[0]))

	photo = Image.open(os.path.join(path_to_photos, photo_filename))
	mask = Image.open(os.path.join(path_to_masks, mask_filename))

	start_time = time.time()
	origina_size = photo.size

	if (photo.size == mask.size):
		photo.save(os.path.join(path_to_resized_folder, photo_filename))
	else:
		photo = photo.resize(mask.size, Image.ANTIALIAS)
		photo.save(os.path.join(path_to_resized_folder, photo_filename))

	if (photo.size != mask.size):
		raise ValueError('Unmatching photo and mask sizes even after resizing: {} vs {}', photo.size, mask.size)

	print 'Resizing of image {} from {} to {} completed in {} sec'.format(photo_filename, origina_size, photo.size, time.time()-start_time)


if __name__ == '__main__':

	photo_files = get_files(PATH_TO_PHOTOS)
	mask_files = get_files(PATH_TO_MASKS)

	if (len(photo_files) != len(mask_files)):
		raise ValueError('Unmatching photo and mask file dataset sizes: {} vs {}'.format(len(photo_files), len(mask_files)))

	num_cores = multiprocessing.cpu_count()
	n_jobs = min(32, num_cores)

	Parallel(n_jobs=n_jobs, backend='threading')(
		delayed(resize)(
			PATH_TO_PHOTOS,
			photo_files[i],
			PATH_TO_MASKS,
			mask_files[i],
			PATH_TO_RESIZED_PHOTOS) for i in range(0, len(photo_files)))
