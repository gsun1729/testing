import math
import numpy as np
from scipy import stats
from skimage.morphology import dilation, disk, erosion

def distance_2d(p0, p1):
	'''
	Returns the euclidian distance between two points in 2d linspace
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

	# Indexes of duplicate points
	failures = []
	# Compute upper triangular of distance matrix
	for x in xrange(0, num_pts):
		for y in xrange(x + 1, num_pts):
			distance_array[x, y] = distance_2d(points_no_R[x], points_no_R[y])
			if distance_array[x, y] != 0 and distance_array[x, y] <= max_sep:
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


def create_dividing_mask(img_mask, collision_pt, erode = 2, dilate = 4):
	'''
	Helper function. Given a list of points, determine the line that cuts through all of them,
	Also given the image to be cut in half, determine which pixels in the image fall on the line.
	Creates a structuring element with two parallel lines
	'''
	x = collision_pt[1]
	y = collision_pt[0]
	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

	structuring_mask = np.zeros_like(img_mask)
	x_d, y_d = structuring_mask.shape
	for x in xrange(x_d):
		for y in xrange(y_d):
			if np.ceil(y * slope + intercept) == x:
				structuring_mask[x, y] = 1
	structuring_mask = dilation(structuring_mask, disk(dilate))
	structuring_mask = erosion(structuring_mask, disk(erode))
	# Take derivative of an image
	sm_derivative = np.zeros_like(img_mask)
	for x in xrange(x_d - 1):
		for y in xrange(y_d - 1):
			if structuring_mask[x + 1, y] - structuring_mask[x, y] != 0:
				sm_derivative[x, y] = 1
	return structuring_mask, sm_derivative


def remove_element_bounds(image, lower_area = 500, upper_area = 3000):
	'''
	accepts an image with segemented binary elements labeled 0-x int and removes
	any that exceed an area greater than upper_area and are under lower_area
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
	return [pt for pt in list_coords if not (pt[0] < 0 or pt[1] < 0)]

test = [(29, 14), (-1, 14), (29, 14), (-1, 14), (14, 29), (14, 29), (14, -1), (14, -1), (29, 15), (-1, 15), (29, 13), (-1, 13), (15, 29), (13, 29), (15, -1), (13, -1), (29, 16), (-1, 16), (29, 12), (-1, 12), (16, 29), (12, 29), (16, -1), (12, -1), (29, 17), (-1, 17), (29, 11), (-1, 11), (17, 29), (11, 29), (17, -1), (11, -1), (28, 18), (0, 18), (28, 10), (0, 10), (18, 28), (10, 28), (18, 0), (10, 0), (28, 19), (0, 19), (28, 9), (0, 9), (19, 28), (9, 28), (19, 0), (9, 0), (28, 20), (0, 20), (28, 8), (0, 8), (20, 28), (8, 28), (20, 0), (8, 0), (27, 21), (1, 21), (27, 7), (1, 7), (21, 27), (7, 27), (21, 1), (7, 1), (27, 22), (1, 22), (27, 6), (1, 6), (22, 27), (6, 27), (22, 1), (6, 1), (26, 23), (2, 23), (26, 5), (2, 5), (23, 26), (5, 26), (23, 2), (5, 2), (25, 24), (3, 24), (25, 4), (3, 4), (24, 25), (4, 25), (24, 3), (4, 3)]

print remove_neg_pts(test)
