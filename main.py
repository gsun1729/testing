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

from scipy import fftpack

def main():
	root = ".\\data\\generated"
	# print get_img_filenames(root)
	cell = io.imread(".\\data\\linhao\\hs\\P34A12_1_w1488 Laser.TIF")
	mito = io.imread(".\\data\\linhao\\hs\\P34A12_1_w2561 Laser.TIF")
	a = max_projection(cell)
	properties(a)
	# a = median(a, disk(5))
	# a = gaussian_filter(a,disk(1.5), mode='constant')
	a = gamma_stabilize(a)
	
	q = robust_binarize(a)
	# view_2d_img(disk_hole(mito[5,:,:],radius = 50, pinhole = True))
	view_2d_img(q)
	properties(q)


	# q = median_layers(cell)
	# stack_viewer(q)
	# Cell outline processing block
	# q = cell[5,:,:]
	# f = np.fft.fft2(q)
	# fshift = np.fft.fftshift(f)
	# magnitude_spectrum = 20*np.log(np.abs(fshift))
	# view_2d_img(magnitude_spectrum)
	# properties(fft2)
	# print dtype2bits[a.dtype.name]
	# c = gamma_stabilize(a)
	# c = median(a, disk(1))
	# properties(c)
	# view_2d_img(c)
	# # 	# c = gamma_stabilize(a)
    # #
	# # c = normalize(c, axis = 0, norm = 'max')
	# # view_2d_img(a)
	# selem = disk(10)
	# d = median(c,selem)
	# view_2d_img(a-c)

	# # view_2d_img(c-a)


if __name__ == "__main__":
	main()
