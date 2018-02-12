import numpy as np
import sys
from skimage.exposure import adjust_gamma
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import median, rank, threshold_otsu
from skimage.segmentation import random_walker
from skimage.restoration import denoise_bilateral, estimate_sigma
dtype2bits = {'uint8': 8,
			  'uint16': 16,
			  'uint32': 32}

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


def fft_ifft(image, struct_disk, pinhole = False):
	fft_transform = np.fft.fft2(image)
	f_shift = np.fft.fftshift(fft_transform)

	f_shift_filtered = f_shift * struct_disk

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
