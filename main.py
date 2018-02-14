import sys

sys.path.insert(0, '.\\lib')
import os

from skimage import io
from render import *
from processing import *
from math_funcs import *
from properties import properties
from read_write import *

from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import (disk, dilation, watershed,
								closing, opening, erosion, skeletonize, medial_axis)
# from skimage.segmentation import random_walker
# from skimage.restoration import denoise_bilateral, estimate_sigma
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
	# Good Images
	cell = io.imread(".\\data\\linhao\\hs\\P26H5_2_w1488 Laser.TIF")
	mito = io.imread(".\\data\\linhao\\hs\\P26H5_2_w2561 Laser.TIF")
	# Bad Images
	cellb = io.imread(".\\data\\linhao\\hs\\P34A12_3_w1488 Laser.TIF")
	mitob = io.imread(".\\data\\linhao\\hs\\P34A12_2_w2561 Laser.TIF")


	# CELL PROCESSING LINE
	sel_elem = disk(2)
	a1 = max_projection(cellb)
	a2 = gamma_stabilize(a1, alpha_clean = 1.3)

	a3 = smooth(a2)
	a4 = median(a3, sel_elem)
	a5 = erosion(a4, selem = disk(1))

	a6 = median(a5, sel_elem)
	a7 = dilation(a6, selem = disk(1))
	a8 = img_type_2uint8(a7, func = 'floor')
	a9 = binarize_image(a8, _dilation = 0, heterogeity_size = 10, feature_size = 2)
	a10 = binary_fill_holes(a9).astype(int)
	# view_2d_img(a9)
	a11 = label_and_correct(a10, a8, min_px_radius = 20)
	a12 = remove_element_bounds(a11, lower_area = 500, upper_area = 3000)
	a13 = measure.find_contours(a12, level = 0.8, fully_connected = 'low', positive_orientation = 'low')

	for item_contour in a13:
		# print
		if item_contour.shape[0] >= 250 and item_contour.shape[0] <= 350:
			holding = points2img(item_contour)
			split_cells = hough_num_circles(holding)




	fig, ax = plt.subplots()
	ax.imshow(a11, interpolation = 'nearest', cmap = plt.cm.gray)
	for n, contour in enumerate(a13):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()

	montage_n_x((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11, a12))

	sys.exit()

	d = disk_hole(a7, 10, pinhole = True)

	# USE ONLY FOR MITOS
	# a8 = fft_ifft(a7, d)

	# ff = binarize_image(a1)

	# montage_n_x((a1,ff))

	# img_type_2uint8(a8)
	b = img_type_2uint8(a7, func = 'floor')
	properties(b)
	c = binarize_image(b)
	d = label_and_correct(c,b,min_px_radius = 20)

	remove_element_bounds(d)
	montage_n_x((c, d))
	# sys.exit()

	e = measure.find_contours(d, 0.8)


	print len(e)
	for x in e:
		if x.shape[0] >= 300 and x.shape[0] <= 350:
			holding = x
			# print holding
			# plot_contour(holding)
			z = points2img(holding)
			q = hough_num_circles(z)
			# montage_n_x((z,q,c))
			# break














	# Display the image and plot all contours found
	fig, ax = plt.subplots()
	ax.imshow(d, interpolation='nearest', cmap=plt.cm.gray)
	# print e
	for n, contour in enumerate(e):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
	# montage_n_x((a1,a2,a3,a4,a5,a6,a7,a8))




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
