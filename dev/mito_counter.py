from lib.render import *
import dev.pathfinder
import numpy as np
import scipy.io
import sys
import argparse
from scipy import ndimage as ndi

kernel2D_connections = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
								 [1, 1, 0, 1, 0, 1, 0, 0],
								 [1, 0, 1, 1, 0, 0, 1, 0],
								 [0, 1, 1, 1, 0, 0, 0, 1],
								 [1, 0, 0, 0, 1, 1, 1, 0],
								 [0, 1, 0, 0, 1, 1, 0, 1],
								 [0, 0, 1, 0, 1, 0, 1, 1],
								 [0, 0, 0, 1, 0, 1, 1, 1]])
path_direction = np.zeros((8, 8), dtype = bool)

def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for image visualization')
	parser.add_argument('-i', dest = 'image', help = 'Image path', required = True)
	options = vars(parser.parse_args())
	return options


def main(args):
	options = get_args(args)
	path = options['image']
	image = scipy.io.loadmat(path)['data']

	z, x, y = image.shape
	image[image > 0] = 1
	stack_viewer(image)
	# for stack_index in xrange(z - 1):
	# 	labeled_stack, object_no = ndi.label(image[stack_index, :,:], structure = np.ones((3, 3)))
	# 	view_2d_img(labeled_field)


def layer_comparator(image3D):
	equivalency_table = []
	zdim, xdim, ydim = image3D.shape
	kernel_dim = 2
	last_used_label = 0
	kernel = np.zeros((2,2,2))
	for z in range(1, zdim):
		for x in xrange(1, xdim):
			for y in xrange(1, ydim):
				print z,x,y
				Query = image3D[z - kernel_dim + 1:z + 1,
								x - kernel_dim + 1:x + 1,
								y - kernel_dim + 1:y + 1].flatten()
				g = dev.pathfinder.Graph()
				g.connections2graph(kernel2D_connections, path_direction, Query)
				
				print Query

				return


if __name__ == "__main__":
	a = np.zeros((10,10))
	a2 = np.zeros((10,10))
	b = np.zeros((10,10))
	a[3:8,4:7] = 1
	a[0,0] = 1
	a[0,1] = 2
	a[1,0] = 4
	a[1,1] = 3
	a2[5:9,6:9] = 1
	b[5:9,6:9] = 1
	a2[0,0] = 21
	a2[0,1] = 22
	a2[1,0] = 24
	a2[1,1] = 23

	# a[0,0] = 0
	# a[0,1] = 0
	# a[1,0] = 0
	# a[1,1] = 0
	# a2[0,0] = 0
	# a2[0,1] = 0
	# a2[1,0] = 0
	# a2[1,1] = 0
	stack = np.array([a,a2,b])
	# print stack.shape
	print stack
	# print b [[[2,2],[3,0]],[[0 ,2],[0,1]]]
	neighbors = stack[0:2,0:2,0:2]
	# q = np.array([1,4,3,5,6,7])
	# print q
	# print list(q[[1,2,3]])
	print kernel2D_connections

	# print test.return_flat()
	# print test.get_1U_neighbor()
	# print test.get_2U_neighbor()
	# print test.get_3U_neighbor()
	# print test.get_POI()




	# q = [i for i in neighbors if i != 0]

	layer_comparator(stack)
	# print [a[item]==0 for item in a]
	# q = all(v == 0 for v in row for row in a)

	# print any(0 in sublist for sublist in q)
