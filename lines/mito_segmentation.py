import sys, os, argparse
import re
import numpy as np
from skimage import io
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter
from skimage.filters import median
from scipy.stats import iqr
from skimage.filters import threshold_local
from scipy import ndimage as ndi
from lib.render import * 
from skimage.morphology import skeletonize_3d


dtype2bits = {'uint8': 8,
			  'uint16': 16,
			  'uint32': 32}


dtype2range = { 'uint8': 255,
				'uint16': 65535,
				'uint32': 4294967295,
				'uint64': 18446744073709551615}


def avg_projection(image, axis = 0):
	'''Axis is defined as the index of the image.shape output.
	By default it is the Z axis (z,x,y)
	Takes a 3d image, and projects it along a defined axis to form a 2d Image
	Takes the average value for each pixel in the (default) x,y plane along the
	z axis

	:param image: [np.ndarray] 3d stack image in <np.ndarray> format
	:param axis: [int] axis to sum along, z = 0, x = 1, y = 2
	:return: [np.ndarray] returns 2d image in the shape x,y
	'''
	try:
		# print axis
		z, x, y = image.shape
		return np.sum(image, axis)//z
	except ValueError:
		if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
			raise Exception("Axis value invalid")
		else:
			raise Exception("Image input faulty")


def max_projection(image, axis = 0):
	'''Axis is defined as the index of the image.shape output.
	By default it is the Z axis (z,x,y)
	Takes a 3d image, and projects it along a defined axis to form a 2d Image
	Takes the max value for each pixel along (default) x,y plane, and projects it
	to one plane

	:param image: [np.ndarray] 3d stack image in <np.ndarray> format
	:param axis: [int] axis to sum along, z = 0, x = 1, y = 2
	:return: [np.ndarray] returns 2d image in the shape x,y
	'''
	try:
		return np.amax(image, axis)
	except ValueError:
		if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
			print("Axis value invalid")
		else:
			print("Image input faulty")
		raise Exception


def gamma_stabilize(image, alpha_clean = 5, floor_method = 'min'):
	"""Normalizes the luma curve. floor intensity becomes 0 and max allowed by the bit number - 1
	Borrowed from Andrei's Imagepipe

	:param image: [np.ndarray]
	:param alpha_clean: [int] size of features that would be removed if surrounded by a majority of
	:param floor_method: [str] ['min', '1q', '5p', 'median'] method of setting the floor intensity. 1q is first quartile, 1p is the first percentile
	:return: [np.ndarray]
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
	

def fft_ifft(image, struct_element):
	'''Performs a fast fourier transform, removes certain frequencies highlighted by
	the structuring element, and returns the inverse fourier transform back.
	Helper function disk_hole

	:param image: [np.ndarray] Image to be filtered
	:param struct_element: [np.ndarray] filter to be applied to image in frequency space, should be same dimension as input image
	:return: [np.ndarray] filtered image
	'''
	# print "> Performing FFT>filter>IFFT transform"

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


def smooth(image, smoothing_px = 0.5, threshold = 1):
	"""Gaussian smoothing of the image
	Borrowed from Andrei's Imagepipe

	:param image: [np.ndarray] Input image
	:param smoothing_px: [float] size of smoothing pixel
	:param threshold: [int] threshold to filter image intensity
	:return: [np.ndarray]
	"""
	# print "> Filtering image with Gaussian filter"
	if len(image.shape) > 2:
		for i in range(0, image.shape[0]):
			image[i, :, :] = gaussian_filter(image[i, :, :],
											 smoothing_px, mode='constant')
			image[image < threshold * np.mean(image)] = 0
	else:
		image = gaussian_filter(image, smoothing_px, mode='constant')
		image[image < threshold * np.mean(image)] = 0
	return image


def disk_hole(image, radius, pinhole = False):
	'''Returns either an image of a pinhole or a circle in the
	middle for removing high frequency/ low frequency noise using FFT
	same dimensions as input image
	Pinhole =  True : pinhole filter, or high pass filter. Filters out low frequency
	content to yield edges
	Pinhole = False: single dot filter, preserves low frequency content

	:param image: [np.ndarray] 2d input image (filter will be applied eventually), used to get dims
	:param radius: [int] radius of pinhole/pinpoint
	:param pinhole: [bool] determines whether the filter will be a pinhole or pinpoint
	:return: [np.ndarray] 2d filter of same size of 2d image input
	'''
	x, y = image.shape
	structuring_element = np.zeros((x, y), dtype = int)
	center = x//2

	for rows in range(x):
		for cols in range(y):
			if (rows - center + 0.5) ** 2 + (cols - center + 0.5) ** 2 <= radius ** 2:
				structuring_element[rows, cols] = 1
	if pinhole:
		return 1 - structuring_element
	else:
		return structuring_element


def img_type_2uint8(base_image, func = 'floor'):
	'''
	Converts a given image type to a uint8 image
	Rounding is done either via 'floor', 'ceiling', or 'fix' functions in numpy

	:param base_image: [np.ndarray] input image
	:param func: [str] function used for scaling image pixel intensity
	:return: [np.ndarray] uint8 image
	'''
	# print "> Converting Image to uin8"
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
		print("Function '{}' not recognized ".format(func))
		raise Exception


def global_max(img_2d):
	'''Returns the maximum pixel value within a 2-3d image'''
	return np.amax(img_2d.flatten())


def global_min(img_2d):
	'''
	Returns the minimum pixel value within a 2-3d image
	'''
	return np.amin(img_2d.flatten())


def label_and_correct(binary_channel, pre_binary, min_px_radius = 10, max_px_radius = 100, min_intensity = 0, mean_diff = 10):
	"""
	Labelling of a binary image, with constraints on minimal feature size, minimal intensity of area
	 covered by a binary label or minimal mean difference from background

	:param binary_channel: [np.ndarray] input image
	:param pre_binary: [np.ndarray] used to compute total intensity
	:param min_px_radius: [float] minimal feature size
	:param min_intensity: [float] minimal total intensity
	:param mean_diff: [float] minimal (multiplicative) difference from the background
	:return: [np.ndarray]
	"""
	labeled_field, object_no = ndi.label(binary_channel, structure = np.ones((3, 3)))

	# prebinary_px = pre_binary.flatten()
	# n_bins = int(2 * iqr(prebinary_px) * (len(prebinary_px) ** (1/3)))
	# n, bin_edge = np.histogram(prebinary_px, n_bins)
	# peak_max_indx = np.argmax(n)
	# background_val = (bin_edge[peak_max_indx] + bin_edge[peak_max_indx + 1]) / 2
	background_mean = np.mean(pre_binary[labeled_field == 0])
	for label in range(1, object_no+1):
		mask = labeled_field == label
		px_radius = np.sqrt(np.sum((mask).astype(np.int8)))
		total_intensity = np.sum(pre_binary[mask])
		label_mean = np.mean(pre_binary[labeled_field == label])
		if px_radius < min_px_radius or total_intensity < min_intensity or label_mean < mean_diff * background_mean or px_radius > max_px_radius:
			labeled_field[labeled_field == label] = 0
	# dbg.label_and_correct_debug(labeled_field)
	return labeled_field


def binarize_img(raw_img):
	'''
	Function reads in a z stack image and returns the segmented binary image in the form of a numpy array.
	'''
	z, x, y = raw_img.shape


	binary = np.zeros_like(raw_img)

	for img_slice in range(z):
		struct_elemt = disk(1)
		slice_data = raw_img[img_slice, :, :]

		output1 = gamma_stabilize(slice_data, alpha_clean = 1, floor_method = 'min')

		output2 = smooth(output1, smoothing_px = 2, threshold = 1)
		median_filtered = median(output1, struct_elemt)
		fft_filter_disk = disk_hole(median_filtered, radius = 5, pinhole = True)
		# Remove High frequency noise from image
		FFT_Filtered = fft_ifft(median_filtered, fft_filter_disk)


		# Convert image to 8 bit for median filter to work
		image_8bit = img_type_2uint8(FFT_Filtered, func = 'floor')
		test = median(image_8bit, struct_elemt)

		test_px_dataset = test.flatten()
		n_bins = int(2 * iqr(test_px_dataset) * (len(test_px_dataset) ** (1/3)))
		n, bin_edge = np.histogram(test_px_dataset, n_bins)
		test_peak_max_indx = np.argmax(n)
		bin_midpt = (bin_edge[test_peak_max_indx] + bin_edge[test_peak_max_indx + 1]) / 2
		test_mask = test > bin_midpt
		test_masked = test * test_mask

		local_thresh = threshold_local(test_masked,
										block_size = 31,
										offset = -15)
		binary_local = test_masked > local_thresh
		# label individual elements and remove really small noise and background
		corrected_slice = label_and_correct(binary_local, test,
												min_px_radius = 1,
												max_px_radius = 100,
												min_intensity = 0,
												mean_diff = 11.9)
		corrected_slice[corrected_slice > 0] = 1
		binary[img_slice, :, :] = corrected_slice

	return binary


def skeletonize_binary(binary_data):
	return skeletonize_3d(binary_data)


def get_args(args):
	parser = argparse.ArgumentParser(description = " Script returns skeletonization, binary, validation images for a given mitochondrial image")

	parser.add_argument('-r', dest = 'read_dir', help = 'read directory for data', required = True)
	parser.add_argument('-w', dest = 'write_dir', help = 'write directory for results', required = True)
	options = vars(parser.parse_args())
	return options


def isfile(path):
	'''
	returns true if path leads to a file, returns false if path leads to a folder, raises error if doesnt exist
	'''
	if not os.path.exists(path):
		raise Exception("{} does not exist".format(path))
	elif os.path.isfile(path):
		return True
	else:
		return False


def get_img_filenames(root_directory, suffix = '.jpg'):
	'''
	Given a root directory, traverses all sub_directories and recovers any files with a given suffix.
	Assigns a UUID to each of the files found

	:param root_directory: [str] location to search
	:param suffix: [str] type of image suffix to look for
	'''
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if (suffix.lower() in img_file.lower()) and '_thumb_' not in img_file:
					img_filename = re.sub(suffix, '', img_file, flags=re.IGNORECASE)
					yield os.path.join(current_location, img_file)


def main(args):
	options = get_args(args)

	# test if file or folder, and read accordingly
	if not isfile(options['read_dir']):
		filepath_data = get_img_filenames(options['read_dir'], suffix = '.tif')
	else:
		filepath_data = [options['read_dir']]


	# execute segmentation
	for filepath in filepath_data:
		raw_img = io.imread(filepath)
		binary_img = binarize_img(raw_img)
		sklton_img = skeletonize_binary(binary_img)
		max_project = max_projection(raw_img)
		max_bproject = max_projection(binary_img)


		raise Exception



if __name__ == "__main__":
	main(sys.argv)