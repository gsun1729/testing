import sys

sys.path.insert(0, '.\\lib')
import os
import re
from PIL import Image
from skimage import io
from render import *
from processing import *
from math_funcs import *
from sklearn.preprocessing import normalize

from scipy import ndimage as ndi, stats
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import median, rank, threshold_otsu
from skimage.morphology import disk, dilation, watershed, closing, skeletonize, medial_axis
from skimage.segmentation import random_walker

def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.tif','', re.IGNORECASE)
					img_filelist.append((img_filename, img_file, os.path.join(current_location, img_file)))
	return img_filelist

def main():
	root = ".\\data\\generated"
	print get_img_filenames(root)
	cell = io.imread(".\\data\\linhao\\hs\\P34A12_1_w1488 Laser.TIF")
	mito = io.imread(".\\data\\linhao\\hs\\P34A12_1_w2561 Laser.TIF")
	print dtype2bits[mito.dtype.name]
	# test = np.uint8(mito)
	# print test
	a = (max_projection(mito, axis = 0))
	print dtype2bits[a.dtype.name]
	c = gamma_stabilize(a)
	# c = median(a, disk(50))
		# c = gamma_stabilize(a)
    #
	# c = normalize(c, axis = 0, norm = 'max')
	# view_2d_img(a)
	view_2d_img(a-c)
	print global_max(a), global_min(a)
	print global_max(c), global_min(c)
	# view_2d_img(c-a)


if __name__ == "__main__":
	main()
