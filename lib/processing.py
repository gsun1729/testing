import numpy as np
import sys
from skimage.exposure import adjust_gamma
from skimage import io
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import median, rank, threshold_otsu
from skimage.segmentation import random_walker
from skimage.restoration import denoise_bilateral, estimate_sigma

from properties import global_max, global_min
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from math_funcs import *
from render import *
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import active_contour
from skimage.filters import gaussian

dtype2bits = {'uint8': 8,
			  'uint16': 16,
			  'uint32': 32}

dtype2range = { 'uint8': 255,
				'uint16': 65535,
				'uint32': 4294967295,
				'uint64': 18446744073709551615}

def gamma_stabilize(image, alpha_clean=5, floor_method='min'):
	"""
	Normalizes the luma curve. floor intensity becomes 0 and max allowed by the bit number - 1

	:param image:
	:param alpha_clean: size of features that would be removed if surrounded by a majority of
	:param floor_method: ['min', '1q', '5p', 'median'] method of setting the floor intensity. 1q is first quartile, 1p is the first percentile
	:return:
	"""
	bits = dtype2bits[image.dtype.name]
	if floor_method == 'min':
		inner_min = np.min(image)
	elif floor_method == '1q':
		inner_min = np.percentile(image, 25)
	elif floor_method == '5p':
		inner_min = np.percentile(image, 5)
	elif floor_method == 'median':
		inner_min = np.median(image)
	else:
		raise PipeArgError('floor_method can only be one of the three types: min, 1q, 5p or median')
	stabilized = (image - inner_min) / (float(2 ** bits) - inner_min)
	stabilized[stabilized < alpha_clean*np.median(stabilized)] = 0
	return stabilized


def sum_projection(image, axis = 0):
	'''
	Axis is defined as the index of the image.shape output.
	By default it is the Z axis (z,x,y)
	'''
	try:
		return np.sum(image, axis)
	except ValueError:
		if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
			print "Axis value invalid"
		else:
			print "Image input faulty"
		sys.exit()


def max_projection(image, axis = 0):
	'''
	Axis is defined as the index of the image.shape output.
	By default it is the Z axis (z,x,y)
	'''
	try:
		return np.amax(image, axis)
	except ValueError:
		if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
			print "Axis value invalid"
		else:
			print "Image input faulty"
		sys.exit()


def avg_projection(image, axis = 0):
	'''
	Axis is defined as the index of the image.shape output.
	By default it is the Z axis (z,x,y)
	'''
	try:
		print axis
		z, x, y = image.shape
		return np.sum(image, axis)//z
	except ValueError:
		if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
			print "Axis value invalid"
		else:
			print "Image input faulty"
		sys.exit()


def disk_hole(image, radius, pinhole = False):
	'''
	Returns either an image of a pinhole or a circle in the
	middle for removing high frequency/ low frequency noise using FFT
	'''
	x, y = image.shape
	structuring_element = np.zeros((x, y), dtype = long)
	center = x//2

	for rows in xrange(x):
		for cols in xrange(y):
			if (rows - center + 0.5) ** 2 + (cols - center + 0.5) ** 2 <= radius ** 2:
				structuring_element[rows, cols] = 1
	if pinhole:
		return 1 - structuring_element
	else:
		return structuring_element


def smooth(image, smoothing_px = 0.5, threshold = 1):
	"""
	Gaussian smoothing of the image

	:param image:
	:param smoothing_px:
	:return:
	"""
	print ">Filtering image with Gaussian filter"
	if len(image.shape) > 2:
		for i in range(0, image.shape[0]):
			image[i, :, :] = gaussian_filter(image[i, :, :],
											 smoothing_px, mode='constant')
			image[image < threshold * np.mean(image)] = 0
	else:
		image = gaussian_filter(image, smoothing_px, mode='constant')
		image[image < threshold * np.mean(image)] = 0
	return image


def fft_ifft(image, struct_element):
	'''
	Performs a fast fourier transform, removes certain frequencies highlighted by
	the structuring element, and returns the inverse fourier transform back.
	Pinhole =  True : pinhole filter, or high pass filter. Filters out low frequency
	content to yield edges
	Pinhole = False: single dot filter, preserves low frequency content
	'''
	print ">Performing FFT>filter>IFFT transform"

	fft_transform = np.fft.fft2(image)
	f_shift = np.fft.fftshift(fft_transform)

	# magnitude_spectrum = 20*np.log(np.abs(f_shift))
	# view_2d_img(magnitude_spectrum)
	# view_2d_img(struct_element)

	f_shift_filtered = f_shift * struct_element

	# magnitude_spectrum_filtered = 20*np.log(np.abs(f_shift_filtered))
	# view_2d_img(magnitude_spectrum_filtered)

	f_inv_shift = np.fft.ifftshift(f_shift_filtered)
	recovered_img = np.fft.ifft2(f_inv_shift)
	recovered_img = np.abs(recovered_img)
	return recovered_img


def bandpass_disk(image, r_range = (10, 200), pinhole = False):
	outer = disk_hole(image, r_range[1], pinhole)
	inner = disk_hole(image, r_range[0], pinhole)
	structuring_element = outer - inner
	return structuring_element


def median_layers(image, struct_disk_r = 5):
	for i in range(0, image.shape[0]):
		image[i, :, :] = median(image[i, :, :], disk(struct_disk_r))
		# image[image < 2 * np.mean(image)] = 0
	return image


def img_type_2uint8(base_image, func = 'floor'):

	print ">Converting Image to uin8"

	try:
		bi_max_val = global_max(base_image)
		bi_min_val = global_min(base_image)
		dt_max = dtype2range['uint8']
		dt_min = 0

		# scaled = dt_min * (1 - ((base_image - bi_min_val) / (bi_max_val - bi_min_val))) + dt_max * ((base_image - bi_min_val)/(bi_max_val - bi_min_val))
		scaled = (base_image - bi_min_val) * ((dt_max - dt_min) / (bi_max_val - bi_min_val)) + dt_min

		if func == 'floor':
			pre_int = np.floor(scaled)
		elif func == 'ceiling':
			pre_int = np.ceil(scaled)
		elif func == 'fix':
			pre_int = np.fix(scaled)
		else:
			raise IOError

		return np.uint8(pre_int)
	except IOError:
		print "Function '{}' not recognized ".format(func)
		sys.exit()


def binarize_image(base_image, _dilation = 0, heterogeity_size = 10, feature_size = 2):
	print ">Binarizing Image..."
	if np.percentile(base_image, 99) < 0.20:
		if np.percentile(base_image, 99) > 0:
			mult = 0.20 / np.percentile(base_image, 99)  # poissonean background assumptions
		else:
			mult = 1000. / np.sum(base_image)
		base_image = base_image * mult
		base_image[base_image > 1] = 1

	clustering_markers = np.zeros(base_image.shape, dtype=np.uint8)
	selem2 = disk(feature_size)
	print '>Performing Local Otsu'
	local_otsu = rank.otsu(base_image, selem2)
	clustering_markers[base_image < local_otsu * 0.9] = 1
	clustering_markers[base_image > local_otsu * 1.1] = 2
	print ">Performing Random Walker Binarization"
	binary_labels = random_walker(base_image, clustering_markers, beta=10, mode='bf') - 1

	if _dilation:
		selem = disk(_dilation)
		binary_labels = dilation(binary_labels, selem)
	return binary_labels


def smooth_contours(binary_group_img):
	'''
	Function is broken as fuck DO NOT USE
	Given a binary image with edges = 1, attempts to smooth out the contours of the individual elements in the image
	'''
	boundary = obtain_border(binary_group_img)
	init = points2img(boundary)

	snake = active_contour(binary_group_img, boundary, alpha = 0.015, beta = 10, gamma = 0.001)

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.imshow(binary_group_img, cmap=plt.cm.gray)
	ax.plot(boundary[:, 0], boundary[:, 1], '--r', lw=3)
	ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
	ax.set_xticks([]), ax.set_yticks([])
	ax.axis([0, binary_group_img.shape[1], binary_group_img.shape[0], 0])
	plt.show()


def hough_num_circles(input_binary_img, min_r = 15, max_r = 35, step = 2):
	'''
	ONLY CAPABLE OF SEPARATING 2 CELLS WILL BE UPDATED TO THREE EVENTUALLY
	'''
	print ">Performing Hough cell splitting"
	# Create a list of radii to test and perform hough transform to recover circle centers (x,y) and radii
	hough_radii = np.arange(min_r, max_r, 2)
	hough_res = hough_circle(input_binary_img, hough_radii)
	accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
										total_num_peaks=3)
	circles = zip(cy, cx, radii);
	# remove any circles too close to each other

	no_duplicates = crop_close(circles, max_sep = 10)

	# HYPOTHETICAL # of cells
	N_cells = len(no_duplicates)
	# view_2d_img(input_binary_img)
	print ">Number cells in subsection: {}".format(N_cells)
	if N_cells > 1:
		# Grow circle size until there is a collision
		collision = False
		while collision == False:
			# Create empty mask
			mask = np.zeros_like(input_binary_img)
			for center_y, center_x, radius in no_duplicates:
				sub_mask = np.zeros_like(input_binary_img)
				# List of coodinates for perimeter of a circle
				circy, circx = circle_perimeter(center_y, center_x, radius)
				no_negs = remove_neg_pts(zip(circy, circx))
				# print zip(circy, circx)
				# Make sure that the circle being masked, anything out of bounds is not masked
				for y, x in no_negs:
					print y, x, sub_mask.shape
					if y < sub_mask.shape[0] and x < sub_mask.shape[1]:
						sub_mask[y, x] = 5
				# view_2d_img(sub_mask)
				mask += dilation(sub_mask, disk(1))
				# Append circles into the empty image to get overview of outline
				# Aka add submask to mask
			# Grow circle radius by 5% per iteration
			for rows in no_duplicates:
				rows[-1] *= 1.1
				rows[-1] = np.int(rows[-1])

			# Determine if there is a collision between circles
			if np.amax(mask.flatten()) >= 10:
				collision = True
				collision_pt = np.where(mask >= 10)
				print collision_pt
		# Create mask to divide both cells
		# print collision_pt
		dm, d_dm = create_dividing_mask(mask, collision_pt)
		# Fill edges to create mask
		filled_cells = binary_fill_holes(input_binary_img).astype(int)
		montage_n_x((mask, dm, d_dm,  mask + input_binary_img, filled_cells, filled_cells * (1 - dm)))
		# view_2d_img(filled_cells * (1 - dm))
		return filled_cells * (1 - dm)
	else:
		# view_2d_img(input_binary_img)
		return binary_fill_holes(input_binary_img).astype(int)

	# Uncomment for visualization
	# montage_n_x((input_binary_img, filled_cells, dm,  filled_cells * (1 - dm)))
	# for center_y, center_x, radius in no_duplicates:
	# 	circy, circx = circle_perimeter(center_y, center_x, radius)
	# 	input_binary_img[circy, circx] = 10
	# view_2d_img(input_binary_img)


def just_label(binary_image):
	labeled_field, object_no = ndi.label(binary_image, structure=np.ones((3, 3)))
	return labeled_field


def label_and_correct(binary_channel, pre_binary, min_px_radius = 10, max_px_radius = 65, min_intensity = 0, mean_diff = 10):
	"""
	Labelling of a binary image, with constraints on minimal feature size, minimal intensity of area
	 covered by a binary label or minimal mean difference from background

	:param binary_channel:
	:param value_channel: used to compute total intensity
	:param min_px_radius: minimal feature size
	:param min_intensity: minimal total intensity
	:param mean_diff: minimal (multiplicative) difference from the background
	:return:
	"""
	labeled_field, object_no = ndi.label(binary_channel, structure=np.ones((3, 3)))
	background_mean = np.mean(pre_binary[labeled_field == 0])
	# print background_mean
	#
	for label in range(1, object_no+1):
		mask = labeled_field == label
		px_radius = np.sqrt(np.sum((mask).astype(np.int8)))
		total_intensity = np.sum(pre_binary[mask])
		label_mean = np.mean(pre_binary[labeled_field == label])
		if px_radius < min_px_radius or total_intensity < min_intensity or label_mean < mean_diff*background_mean or px_radius > max_px_radius:
			labeled_field[labeled_field == label] = 0
	# dbg.label_and_correct_debug(labeled_field)
	return labeled_field


def cell_split(input_img, contours):
	print ">Starting Cell Split"
	output = np.zeros_like(input_img)
	output[input_img > 0] = 1
	for item_contour in contours:
		# remove cells that have a low circumference or too high circumference
		contour_holding = points2img(item_contour)

		if item_contour.shape[0] >= 100 and item_contour.shape[0] <= 350:
			holding = points2img(item_contour)
			holding_fill = binary_fill_holes(holding).astype(int)
			if sum(holding_fill.flatten()) > 100:
				split_cells = hough_num_circles(holding)
				tlx, tly, brx, bry = location(item_contour)
				# print location(item_contour)
				# print split_cells.shape, output.shape
				for x in xrange(tlx, brx + 1):
					for y in xrange(tly, bry + 1):
						output[y, x] = output[y, x] * split_cells[y - tly, x - tlx]
	return label_and_correct(output, input_img)
