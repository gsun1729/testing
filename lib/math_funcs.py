import math
import numpy as np

def distance_2d(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def crop_close(points, max_sep = 5):
	points_noR = [x[:-1] for x in points]
	num_pts = len(points_noR)
	distance_array = np.zeros((num_pts, num_pts))

	for x in xrange(num_pts):
		for y in xrange(num_pts):
			distance_array[x, y] = distance_2d(points_noR[x], points_noR[y])
	failures = []
	for x in xrange(0, num_pts):
		for y in xrange(x + 1, num_pts):
			if distance_array[x, y] != 0 and distance_array[x, y] <= max_sep:
				failures.append(x)
	failures = sorted(failures, key = int, reverse = True)

	for f in failures:
		del points[f]
	return [list(elem) for elem in points]
