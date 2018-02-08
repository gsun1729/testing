import sys

sys.path.insert(0, '.\\lib')
import os

from skimage import io
from render import *
from processing import *
from math_funcs import *
from properties import properties
from read_write import *

from sklearn.preprocessing import normalize

from scipy import ndimage as ndi, stats
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import median, rank, threshold_otsu
from skimage.morphology import disk, dilation, watershed, closing, skeletonize, medial_axis
from skimage.segmentation import random_walker



def main():
	root = ".\\data\\generated"
	# print get_img_filenames(root)
	cell = ".\\data\\linhao\\hs\\P34A12_1_w1488 Laser.TIF"
	mito = ".\\data\\linhao\\hs\\P34A12_1_w2561 Laser.TIF"
	
	a = (max_projection(mito, axis = 0))
	view_2d_img(a)
	properties(a)
	# print dtype2bits[a.dtype.name]
	# c = gamma_stabilize(a)
	c = median(a, disk(1))
	properties(c)
	view_2d_img(c)
	# 	# c = gamma_stabilize(a)
    # #
	# # c = normalize(c, axis = 0, norm = 'max')
	# # view_2d_img(a)
	# selem = disk(10)
	# d = median(c,selem)
	# view_2d_img(a-c)

	# # view_2d_img(c-a)


if __name__ == "__main__":
	main()
