
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
from skimage.morphology import skeletonize_3d
from collections import defaultdict
import mahotas as mh

import lib.pathfinder as pathfinder
from lib.read_write import *
from lib.render import *

dtype2bits = {'uint8': 8,
			  'uint16': 16,
			  'uint32': 32}


dtype2range = { 'uint8': 255,
				'uint16': 65535,
				'uint32': 4294967295,
				'uint64': 18446744073709551615}

projection_suffix = "_P"
binary_suffix = "_bin"
skel_suffix = "_skel"
img_suffix = "_fig.png"


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
		
		if n_bins <= 0:
			n, bin_edge = np.histogram(test_px_dataset)
		else:
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
	# return skeletonize_3d(binary_data)
	return mh.thin(binary_data)

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


def imglattice2graph(input_binary):
	'''Converts a 3d image into a graph for segmentation

	:param input_binary: [np.ndarray] complete binary image 3d
	:return item_id: [np.ndarray] indicies of all elements in the lattice for identification
	:return graph_map: [graph object] graph object indicating which voxels are connected to which voxels
	'''
	zdim, xdim, ydim = input_binary.shape
	# Instantiate graph
	graph_map = pathfinder.Graph()
	# Create an array of IDs
	item_id = np.array(range(0, zdim * xdim * ydim)).reshape(zdim, xdim, ydim)
	# Traverse input binary image
	# print("\tSlices Analyzed: ",)
	for label in set(input_binary.flatten()):
		if label != 0:
			label_locations = [tuple(point) for point in np.argwhere(input_binary == label)]
			for location in label_locations:
				# Get Query ID Node #
				query_ID = item_id[location]
				# Get neighbors to Query
				neighbor_locations = get_3d_neighbor_coords(location, input_binary.shape)
				# For each neighbor
				for neighbor in neighbor_locations:
					# Get Neighbor ID
					neighbor_ID = item_id[neighbor]
					# If query exists and neighbor exists, branch query and neighbor.
					# If only Query exists, branch query to itself.
					if input_binary[neighbor]:
						graph_map.addEdge(origin = query_ID,
											destination = neighbor_ID,
											bidirectional = False,
											self_connect = True)
					else:
						graph_map.addEdge(origin = query_ID,
											destination = query_ID,
											bidirectional = False,
											self_connect = True)
		else:
			pass
	return item_id, graph_map


def get_3d_neighbor_coords(tuple_location, size):
	'''Gets neighbors directly adjacent to target voxel. 1U distance max. Does not include diagonally adjacent neighbors

	:param tuple_location: [tuple] query location
	:param size: [tuple] size dimensions of the original image listed in order of Z, X, Y, to get rid of any points that exceed the boundaries of the rectangular prism space
	:return: [list] of [tuple] list of tuples indicating neighbor locations
	'''
	neighbors = []
	z, x, y = tuple_location
	zdim, xdim, ydim = size

	top = (z + 1, x, y)
	bottom = (z - 1, x, y)
	front = (z, x + 1, y)
	back = (z, x - 1, y)
	left = (z, x, y - 1)
	right = (z, x, y + 1)

	neighbors = [top, bottom, front, back, left, right]
	neighbors = [pt for pt in neighbors if (pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0) and (pt[0] < zdim and pt[1] < xdim and pt[2] < ydim)]

	return neighbors


def layer_comparator(image3D):
	'''Uses lattice graph data to determine where the unique elements are and prune redundancies.

	:param image3D: [np.ndarray] original binary image 3d
	:return: [np.ndarray] segmented 3d image
	'''
	ID_map, graph = imglattice2graph(image3D)

	graph_dict = graph.get_self()
	# for key in sorted(graph_dict.iterkeys()):
	# 	print("%s: %s" % (key, graph_dict[key]))
	network_element_list = []
	print("> Network size: ", len(graph_dict))
	# print(graph_dict)
	print("> Pruning Redundancies")
	for key in list(graph_dict.keys()):
		try:
			network = sorted(graph.BFS(key))
			for connected_key in network:
				graph_dict.pop(connected_key, None)
			if network not in network_element_list:
				network_element_list.append(network)
		except:
			pass
	print("> Unique Paths + Background [1]: ", len(network_element_list))

	img_dimensions = ID_map.shape
	output = np.zeros_like(ID_map).flatten()

	last_used_label = 1
	print("> Labeling Network")
	for network in network_element_list:
		for element in network:
			output[element] = last_used_label
		last_used_label += 1
	return output.reshape(img_dimensions)


def stack_stack_multply(stack1, stack2):
	'''	Multiplies a 3d stack layer by layer with another 3d stack (hadamard product)

	:param stack1: [np.ndarray] first stack image
	:param stack2: [np.ndarray] second stack image
	:return: composite [np.ndarray] multiplied image
	'''
	z1,x1,y1 = stack1.shape
	z2,x2,y2 = stack2.shape
	if z1 == z2 and x1 == x2 and y1 == y2:
		composite = np.zeros_like(stack1)
		for layer in range(z1):
			composite[layer, :, :] = stack1[layer, :, :] *  stack2[layer, :, :]
	else:
		raise Exception('stack stack dimensions do not match')
	return composite


def stack_multiplier(image, stack):
	'''Multiplies each layer of a 3d stack image (3d image) with a 2d image after
	verifying shape fit

	:param image: [np.ndarray] 2d Image to be multiplied
	:param stack: [np.ndarray] 3d stack image to have 2d image convoluted w/ along all slices
	:return: [np.ndarray] returns a convoluted 3d image
	'''
	z, x, y = stack.shape
	composite = np.zeros_like(stack)
	for layer in range(z):
		composite[layer, :, :] = stack[layer, :, :] * image
	return composite


def get_args(args):
	parser = argparse.ArgumentParser(description = " Script returns skeletonization, binary, validation images for a given mitochondrial image")

	parser.add_argument('-r', dest = 'read_dir', help = 'read directory for data', required = True)
	parser.add_argument('-w', dest = 'write_dir', help = 'write directory for results', required = True)
	options = vars(parser.parse_args())
	return options


def spinning_disk_correction(stack_image):
	'''
	Special function designed to get rid of image abberation on spinning disk microscope 
	Removes two bright spots on the image owing to some strange particulate matter
	'''
	z, x, y = stack_image.shape
	screen_mask = np.ones((x, y))
	screen_mask[40:70, 390:410] = 0
	screen_mask[140:150, 284:293] = 0
	return stack_multiplier(screen_mask, stack_image)


def main(args):
	options = get_args(args)

	# test if file or folder, and read accordingly
	if not isfile(options['read_dir']):
		filepath_data = get_img_filenames(options['read_dir'], suffix = '.tif')
	else:
		filepath_data = [options['read_dir']]


	# execute segmentation
	n = 0
	for filepath in filepath_data:
		file_ID = os.path.splitext(os.path.basename(filepath))[0]
		if 'w1488' in file_ID:
			# file_ID = full_filename.replace("_RAW", "")

			print(file_ID, n)
			raw_img = io.imread(filepath)
			
			binary_img = binarize_img(raw_img)
			binary_img = spinning_disk_correction(binary_img)
			
			labeled_binary = layer_comparator(binary_img)
			skeletonization = skeletonize_binary(binary_img)
			stack_viewer(skeletonization)
			raise Exception
			labeled_skeletons = stack_stack_multply(skeletonization, labeled_binary)

			max_project = max_projection(raw_img)
			max_bproject = max_projection(binary_img)
			max_skelproject = max_projection(skeletonization)

			save_figure(max_project, file_ID + projection_suffix + img_suffix, options['write_dir'])
			save_figure(max_bproject, file_ID + binary_suffix + projection_suffix + img_suffix, options['write_dir'])
			save_figure(max_skelproject, file_ID + skel_suffix + projection_suffix + img_suffix, options['write_dir'])
			save_data(labeled_skeletons, file_ID + skel_suffix, options['write_dir'])
			save_data(labeled_binary, file_ID + binary_suffix, options['write_dir'])
		else:
			pass
		n += 1


if __name__ == "__main__":
	main(sys.argv)