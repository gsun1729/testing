import numpy as np
import sys
from skimage.exposure import adjust_gamma
from skimage import io
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import median, rank, threshold_otsu
from skimage.segmentation import random_walker
from skimage.restoration import denoise_bilateral, estimate_sigma
from render import view_2d_img
from properties import global_max, global_min
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
	if np.percentile(base_image, 99) < 0.20:
		if np.percentile(base_image, 99) > 0:
			mult = 0.20 / np.percentile(base_image, 99)  # poissonean background assumptions
		else:
			mult = 1000. / np.sum(base_image)
		base_image = base_image * mult
		base_image[base_image > 1] = 1
	clustering_markers = np.zeros(base_image.shape, dtype=np.uint8)
	selem2 = disk(feature_size)
	print 'local'
	local_otsu = rank.otsu(base_image, selem2)
	print 'local done'
	clustering_markers[base_image < local_otsu * 0.9] = 1
	clustering_markers[base_image > local_otsu * 1.1] = 2
	print "before rw"
	binary_labels = random_walker(base_image, clustering_markers, beta=10, mode='bf') - 1
	print "post rw"

	if _dilation:
		selem = disk(_dilation)
		binary_labels = dilation(binary_labels, selem)


	return binary_labels


def label_and_correct(binary_channel, pre_binary, min_px_radius = 10, min_intensity = 0, mean_diff = 10):
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
	print background_mean
	#
	for label in range(1, object_no+1):
	    mask = labeled_field == label
	    px_radius = np.sqrt(np.sum((mask).astype(np.int8)))
	    total_intensity = np.sum(pre_binary[mask])
	    label_mean = np.mean(pre_binary[labeled_field == label])
	    if px_radius < min_px_radius or total_intensity < min_intensity or label_mean < mean_diff*background_mean:
	        labeled_field[labeled_field == label] = 0
	# dbg.label_and_correct_debug(labeled_field)
	return labeled_field
