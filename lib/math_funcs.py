import math
import numpy as np
from scipy import stats
from skimage.morphology import dilation, disk, erosion

def distance_2d(p0, p1):
	'''
	Returns the euclidian distance between two points in 2d linspace
	
	:param p0: point 1 tuple
	:param p1: point 2 tuple
	:return: <float> distance between p1 and p0
	'''
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def crop_close(points, max_sep = 20):
	'''
	Given a list of 2d points, removes any possible duplicate points within a
	distance of max_sep
	'''
	points_no_R = [x[:-1] for x in points]
	num_pts = len(points_no_R)
	distance_array = np.zeros((num_pts, num_pts))
	# print points_no_R
	# Indexes of duplicate points
	failures = []
	# Compute upper triangular of distance matrix
	for x in xrange(0, num_pts):
		for y in xrange(x + 1, num_pts):
			distance_array[x, y] = distance_2d(points_no_R[x], points_no_R[y])
			if distance_array[x, y] <= max_sep:
				failures.append(x)
	failures = sorted(failures, key = int, reverse = True)
	# Remove failures
	if len(failures) == len(points_no_R):
		return [list(points_no_R[0])]
	else:
		for f in failures:
			del points[f]
	return [list(elem) for elem in points]


def obtain_border(input_image_2d):
	'''
	returns a list of points that classifies the border of the Image
	Used for defining border around segmented image in active contour finding function

	Returns a Numpy array in the format
	[[1,2]
	 [1,2]
	 [3,2]
	 [4,3]]]
	Helper function for processing.smooth_contours
	Smooth contours is gimmicky as fuck though so don't use it for it
	'''
	points = []
	x_dim, y_dim = input_image_2d.shape
	for x in xrange(x_dim):
		for y in xrange(y_dim):
			if (x == 0 or x == x_dim - 1) or (y == 0 or y == y_dim - 1):
				points.append([x,y])
	return np.asarray(points)

# FUNCTION NO LONGER NEEDED, FOUND BETTER OPTION FUCK YEAH
# def create_dividing_mask(img_mask, collision_pt, erode = 2, dilate = 4):
# 	'''
# 	Helper function. Given a list of points, determine the line that cuts through all of them,
# 	Also given the image to be cut in half, determine which pixels in the image fall on the line.
# 	Creates a structuring element with two parallel lines
# 	'''
# 	x = collision_pt[1]
# 	y = collision_pt[0]
# 	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# 	print x,y
# 	print slope, intercept
# 	# print slope, intercept
# 	structuring_mask = np.zeros_like(img_mask)
# 	x_d, y_d = structuring_mask.shape
# 	if slope == 0:
# 		slope, intercept, r_value, p_value, std_err = stats.linregress(y, x)
# 		for x in xrange(x_d):
# 			for y in xrange(y_d):
# 				if np.ceil(y * slope + intercept) == x:
# 					structuring_mask[x, y] = 1
# 	else:
# 		for x in xrange(x_d):
# 			for y in xrange(y_d):
# 				if np.ceil(x * slope + intercept) == y:
# 					structuring_mask[x, y] = 1
# 	structuring_mask = dilation(structuring_mask, disk(dilate))
# 	structuring_mask = erosion(structuring_mask, disk(erode))
# 	# Take derivative of an image
# 	sm_derivative = np.zeros_like(img_mask)
# 	for x in xrange(x_d - 1):
# 		for y in xrange(y_d - 1):
# 			if structuring_mask[x + 1, y] - structuring_mask[x, y] != 0:
# 				sm_derivative[x, y] = 1
# 	return structuring_mask, sm_derivative


def remove_element_bounds(image, lower_area = 500, upper_area = 3000):
	'''
	accepts an image with segemented binary elements labeled 0-x int and removes
	any that exceed an area greater than upper_area and are under lower_area
	Counts the # of pixels labeled with 0-x int and if lower than lower_area,
	or higher than upper_area, removes from the image

	:param image: labeled segmented binary 2d Image
	:param lower_area: minimum pixel area acceptable
	:param upper_area: maximum pixel area acceptable
	:return: <np.ndarray> 2d array with filtered segmented binary labels (not renumbered)
	'''
	max_elements = np.amax(image)
	# sum(float(num) >= 1.3 for num in mylist)

	for element_num in range(1, max_elements + 1):

		area = sum(int(num) == element_num for num in image.flatten())
		# print element_num, area
		if area <= lower_area or area >= upper_area:
			image[image == element_num] = 0

	return image


def remove_neg_pts(list_coords):
	'''
	Given a list of 2d coordinates in the format [(1,2), (2,3),...], removes any
	coordinates with a negative value for x or y

	:param list_coords: list of coordinates in the form of a list of tuples
	:return: list of tuples 2d coordinates
	'''
	return [pt for pt in list_coords if not (pt[0] < 0 or pt[1] < 0)]


def array_all_ones(array):
	'''
	Function determines if a 2d array is full of only ones, used for Hough cell
	splitter

	:param array: 2d numpy array
	:return: <bool> returns whether if >array< contains only ones
	'''
	x, y = array.shape
	result = True
	for xd in xrange(x):
		for yd in xrange(y):
			if array[xd, yd] != 1:
				result = False
				break
				break
	return result


def verify_shape(img_2d, stack_3d):
	'''
	Function verifies that the shape of a 2d image matches with a single slice
	of a 3d stack image. Helper function for stack_multiplier

	:param img_2d: 2d image input
	:param stack_3d: 3d stack image input
	:return: <bool> indicating whether a single slice of the 3d stack matches in dimension w/ the 2d image
	'''
	z3, x3, y3 = stack_3d.shape
	x2, y2 = img_2d.shape
	if x2 == x3 and y2 == y3:
		return True
	else:
		return False


def stack_multiplier(image, stack):
	'''
	Multiplies each layer of a 3d stack image (3d image) with a 2d image after
	verifying shape fit

	:param image: 2d Image to be multiplied
	:param stack: 3d stack image to have 2d image convoluted w/ along all slices
	:return: <np.ndarray> returns a convoluted 3d image
	'''
	z, x, y = stack.shape
	composite = np.zeros_like(stack)
	if verify_shape(image, stack):
		for layer in xrange(z):
			composite[layer, :, :] = stack[layer, :, :] * image
	return composite

# test = [(29, 14), (-1, 14), (29, 14), (-1, 14), (14, 29), (14, 29), (14, -1), (14, -1), (29, 15), (-1, 15), (29, 13), (-1, 13), (15, 29), (13, 29), (15, -1), (13, -1), (29, 16), (-1, 16), (29, 12), (-1, 12), (16, 29), (12, 29), (16, -1), (12, -1), (29, 17), (-1, 17), (29, 11), (-1, 11), (17, 29), (11, 29), (17, -1), (11, -1), (28, 18), (0, 18), (28, 10), (0, 10), (18, 28), (10, 28), (18, 0), (10, 0), (28, 19), (0, 19), (28, 9), (0, 9), (19, 28), (9, 28), (19, 0), (9, 0), (28, 20), (0, 20), (28, 8), (0, 8), (20, 28), (8, 28), (20, 0), (8, 0), (27, 21), (1, 21), (27, 7), (1, 7), (21, 27), (7, 27), (21, 1), (7, 1), (27, 22), (1, 22), (27, 6), (1, 6), (22, 27), (6, 27), (22, 1), (6, 1), (26, 23), (2, 23), (26, 5), (2, 5), (23, 26), (5, 26), (23, 2), (5, 2), (25, 24), (3, 24), (25, 4), (3, 4), (24, 25), (4, 25), (24, 3), (4, 3)]
#
# print remove_neg_pts(test)
#


# a = np.random.rand(5,5)
# z = np.zeros_like(a)
#
# for x in xrange(0, 5):
# 	for y in xrange(x + 1, 5):
# 		z[x,y] = a[x,y]
# print a
# print z
