
import sys
sys.path.insert(0, '.\\lib')
sys.path.insert(0, '.\\lines')
import os
import cell_line
import mito_line
from skimage.morphology import skeletonize_3d
from render import *
# from render import *
# from processing import *
# from math_funcs import *
# from properties import properties
# from read_write import *
#
# from skimage import measure
# from scipy.ndimage.morphology import binary_fill_holes
# from skimage.morphology import (disk, dilation, watershed,
# 								closing, opening, erosion, skeletonize, medial_axis)
# # from skimage.segmentation import random_walker
# # from skimage.restoration import denoise_bilateral, estimate_sigma
# import scipy.signal as ss
# from sklearn.preprocessing import normalize
#
# from scipy import ndimage as ndi, stats
# from scipy.ndimage import gaussian_filter
# from skimage.feature import peak_local_max
# from skimage.filters import median, rank, threshold_otsu, laplace
# from math_funcs import *

def main():
	os.system('cls' if os.name == 'nt' else 'clear')
	root = ".\\data\\generated"
	# print get_img_filenames(root)
	cell = ".\\data\\hs\\P11B3_2_w1488 Laser.TIF"
	mito = ".\\data\\hs\\P11B3_2_w2561 Laser.TIF"
	cell2 = ".\\data\\_hs\\P45F12_3_w1488 Laser.TIF"
	mito2 = ".\\data\\_hs\\P45F12_3_w2561 Laser.TIF"
	cell3 = ".\\data\\hs\\P26G1_1_w1488 Laser.TIF"
	mito3 = ".\\data\\hs\\P26G1_1_w2561 Laser.TIF"
	cell4 = ".\\data\\_hs\\P42A12_1_w1488 Laser.TIF"
	mito4 = ".\\data\\_hs\\P42A12_1_w2561 Laser.TIF"
	# # Bad Images
	# cellb = ".\\data\\hs\\P34A12_3_w1488 Laser.TIF"
	# mitob = ".\\data\\hs\\P34A12_2_w2561 Laser.TIF"

	# cell_line.analyze(cell3)
	binary_map = mito_line.analyze(mito)
	skeleton3d = skeletonize_3d(binary_map)
	stack_viewer(skeleton3d)
	sys.exit()
	# d = disk_hole(a7, 10, pinhole = True)

	# USE ONLY FOR MITOS
	# a8 = fft_ifft(a7, d)

	# ff = binarize_image(a1)

	# montage_n_x((a1,ff))

	# img_type_2uint8(a8)
	b = img_type_2uint8(a7, func = 'floor')
	properties(b)
	c = binarize_image(b)
	d = label_and_correct(c,b, min_px_radius = 20)

	remove_element_bounds(d)
	montage_n_x((c, d))
	# sys.exit()

	e = measure.find_contours(d, 0.8)










	# montage_2x((a, b, b1, c, d), (a,b,c))

	# q = fft_ifft(a, 175, pinhole = False)
	# c = fft_ifft(a, 200, pinhole = False)
	# z = fft_bandpass(a,r_range = (100,200),pinhole = False)
	# sz = fft_bandpass(a,pinhole = True)
	# montage_x((c,q,q-c,z,sz))
	# print save_file(root, root, root, root)
	# sigma_est = estimate_sigma(a, multichannel=False, average_sigmas=True)
	# denoise_bilateral(a,sigma_color=0.1, sigma_spatial=15,
	#             multichannel=False)
	# print sigma_est

	# z = median(a, disk(5))
	# montage_x((a,z))
	# a = gaussian_filter(a,disk(1.5), mode='constant')
	# a = gamma_stabilize(a)

	# q = robust_binarize(a)
	# view_2d_img(disk_hole(mito[5,:,:],radius = 50, pinhole = True))
	# view_2d_img(q)
	# properties(q)


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
