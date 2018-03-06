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


def corner_locations(dimension_tuple):
	'''
	Given n dimensions, returns the coordinates of where the corners should be in the given space in the form of a list of lists

	:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
	:return: list of lists with sublists containing coordinates of the corner space
	'''
	corners = []
	for x in xrange(2 ** len(dimension_tuple)):
		binary = str(np.binary_repr(x))
		if len(binary) < len(dimension_tuple):
			binary = str(0) * (len(dimension_tuple) - len(binary)) + binary
		empty_corner = list(np.zeros(len(binary), dtype = int))
		for coord in xrange(len(binary)):
			empty_corner[coord] = int(binary[coord], 2) * int(dimension_tuple[coord] - 1)
		corners.append(tuple(empty_corner))
	return corners


def edge_locations(dimension_tuple):
	'''
	Hardcoded edge detection
	num edge elements in an ndimensional element = (4 * (np.sum(dimension_tuple) - (2 * len(dimension_tuple)))):
	Can only interpret edges in a 2d or 3d volume.

	:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
	:return: list of lists with sublists containing coordinates of the edges
	'''
	edges = []
	if len(dimension_tuple) == 2:
		x_dim, y_dim = dimension_tuple
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if (x == 0 or x == x_dim - 1) or (y == 0 or y == y_dim - 1):
					edges.append((x, y))

	elif len(dimension_tuple) == 3:
		z_dim, x_dim, y_dim = dimension_tuple
		for z in xrange(z_dim):
			for x in xrange(x_dim):
				for y in xrange(y_dim):
					if (x == 0 or x == x_dim - 1):
						if (y == 0 or y == y_dim - 1):
							edges.append((z, x, y))
					if (x == 0 or x == x_dim - 1):
						if (z == 0 or z == z_dim - 1):
							edges.append((z, x, y))
					if (y == 0 or y == y_dim - 1):
						if (z == 0 or z == z_dim - 1):
							edges.append((z, x, y))
	corners = corner_locations(dimension_tuple)
	edges = [edge for edge in edges if edge not in corners]
	return edges


def face_locations(dimension_tuple):
	faces = []
	if len(dimension_tuple) == 2:
		print "Faces don't exist for 2D geometries"
		return faces
	elif len(dimension_tuple) == 3:
		z_dim, x_dim, y_dim = dimension_tuple
		for z in xrange(z_dim):
			for x in xrange(x_dim):
				for y in xrange(y_dim):
					if (z == 0 or z == z_dim - 1):
						faces.append((z, x, y))
					if (x == 0 or x == x_dim - 1):
						faces.append((z, x, y))
					if (y == 0 or y == y_dim - 1):
						faces.append((z, x, y))
	corners = corner_locations(dimension_tuple)
	edges = edge_locations(dimension_tuple)
	faces = [face for face in faces if face not in corners and face not in edges]
	return faces



def core_locations(dimension_tuple):
	cores = []
	if len(dimension_tuple) == 2:
		x_dim, y_dim = dimension_tuple
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if (x != 0 or x != x_dim - 1) or (y != 0 or y != y_dim - 1):
					cores.append((x, y))

	elif len(dimension_tuple) == 3:
		z_dim, x_dim, y_dim = dimension_tuple
		for z in xrange(z_dim):
			for x in xrange(x_dim):
				for y in xrange(y_dim):
					if (x != 0 or x != x_dim - 1) :
						if (y != 0 or y != y_dim - 1):
							cores.append((z, x, y))
					if (x != 0 or x != x_dim - 1) :
						if (z != 0 or z != z_dim - 1):
							cores.append((z, x, y))
					if (y != 0 or y != y_dim - 1) :
						if (z != 0 or z != z_dim - 1):
							cores.append((z, x, y))
	corners = corner_locations(dimension_tuple)
	edges = edge_locations(dimension_tuple)
	faces = face_locations(dimension_tuple)
	cores = [core for core in cores if core not in corners and core not in edges and core not in faces]
	return cores




def lattice2graph(input_binary):
	dimensions = input_binary.shape
	elements = np.arange(np.prod(dimensions))
	elements = [str(index) for index in elements]
	for x, index in enumerate(elements):
		if len(index) < 3:
			elements[x] = "0" * (3 - len(index)) + index


	if len(dimensions) == 2:
		xdim, ydim = dimensions
		print "2d"
	elif len(dimensions) == 3:
		zdim, xdim, ydim = dimensions
		print "3d"
		test = np.zeros_like(input_binary)

		print test
		n = 0
		print elements
		for z in xrange(zdim):
			for x in xrange(xdim):
				for y in xrange(ydim):
					# Corner Cases
					pass
					# if x == 0 and y == 0 and z == 0:


	else:
		print "Dimensions > 3"

def layer_comparator(image3D):
	equivalency_table = []
	zdim, xdim, ydim = image3D.shape
	kernel_dim = 2
	last_used_label = 1
	kernel = np.zeros((kernel_dim, kernel_dim, kernel_dim))
	kernel_IDs = range(0, kernel_dim ** 3)
	for z in xrange(1, zdim):
		for x in xrange(1, xdim):
			for y in xrange(1, ydim):
				print z,x,y
				Query_kernel = image3D[z - kernel_dim + 1:z + 1,
								x - kernel_dim + 1:x + 1,
								y - kernel_dim + 1:y + 1].flatten()
				print Query_kernel

				if any(item != 0  for item in Query_kernel):
					print "ITEM PRESENT IN KERNEL"
					queryk_exists = np.zeros_like(Query_kernel)
					queryk_exists[Query_kernel > 0] = 1
					kernel_graph = dev.pathfinder.Graph()
					kernel_graph.connections2graph(kernel2D_connections, path_direction, queryk_exists)
					network_element_list = []
					# Determine the number of independent paths within a network
					for ID in kernel_IDs:
						network = kernel_graph.BFS(ID)
						if network:
							if sorted(network) not in network_element_list:
								network_element_list.append(sorted(network))
					# For each independent path, get labels
					for network in network_element_list:
						print network
				else:
					pass



				# Normalize query existance to binary

				# print queryk_exists
				# Create graph of connections

				# If a query has nothing there:

				# if Query_kernel[query_ID] == 0:
				# 	connections2Query = [connection for connection in g.BFS(query_ID) if connection != query_ID]
				# 	# If the query still has neighbors
				# 	if connections2Query:
				#
				# else:
				# 	print "query present"
				# 	# Remove self from list of connections (any node is connected to itself in this context)
				# 	# also get a list of locations of neighbors
				# 	connections2Query = [connection for connection in g.BFS(query_ID) if connection != query_ID]
				# 	neighbor_vals = Query_kernel[connections2Query]
				# 	last_used_label += 1
				# 	# print connections2Query
				# 	# print neighbor_vals
				# 	# print last_used_label
				# 	neighbor_vals = [last_used_label for element in neighbor_vals]
				# 	# print neighbor_vals
				# 	Query_kernel[connections2Query] = neighbor_vals
				# 	# print Query_kernel
				# 	Query_kernel[query_ID] = last_used_label
				# 	print Query_kernel.reshape(2,2,2)
				# 	image3D[z - kernel_dim + 1:z + 1,
				# 					x - kernel_dim + 1:x + 1,
				# 					y - kernel_dim + 1:y + 1] = Query_kernel.reshape(2,2,2)
					# print image3D


				# 		# get a list of the neighbor values
				#
				# 		lowest_neighbor = np.min(neighbor_vals)
				#
				# 		# generate a comprehensive list of equivalencies
				# 		equivalencies = [(n1, n2) for n1 in neighbor_vals for n2 in neighbor_vals]
				# 		# if the rule does not exist, add it to the complete equivalency_table
				# 		for rule in equivalencies:
				# 			if rule in equivalency_table or tuple(reversed(rule)) in equivalency_table or rule[0] == rule[1]:
				# 				pass
				# 			else:
				# 				equivalency_table.append(rule)
				#
				# 		print "neighboar vals", neighbor_vals
				# 		print "dumb neighbor", lowest_neighbor
				# 		last_used_label += 1
				# 		image3D[z, x, y] = last_used_label
				# 	else:
				# 		last_used_label += 1
				# 		image3D[z, x, y] = last_used_label
				# print "last used label\t",last_used_label

			# print

				return


if __name__ == "__main__":
	a = np.zeros((10,10))
	a2 = np.zeros((10,10))
	b = np.zeros((10,10))
	a[3:8,4:7] = 1
	a[0,0] = 1
	a[0,1] = 0
	a[1,0] = 0
	a[1,1] = 1
	a2[5:9,6:9] = 1
	b[5:9,6:9] = 1
	a2[0,0] = 0
	a2[0,1] = 1
	a2[1,0] = 1
	a2[1,1] = 0
	a2[7:9,0:2] =1
	b[7:9,0:2] =1

	a[0:2,7:9] = 1

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
	# print stack
	# lattice2graph(stack)
	test_stack = np.zeros((6, 4, 7))
	# print test_stack.shape
	# print test_stack
	edges = edge_locations(test_stack.shape)
	corners = corner_locations(test_stack.shape)
	cores = core_locations(test_stack.shape)
	faces = face_locations(test_stack.shape)
	for item in faces:
		test_stack[item] = 1
		# test_stack[item[0],item[1]] = 1
	print test_stack

	# core_locations(test_stack.shape)
	# print "asdfasdf"
	# neighbors = stack[0:2,0:2,0:2]
	# n  = np.array([[[2,2],[3,0]],[[7 ,2],[9,1]]])
	# print n
	# print n.flatten()
	# print n.reshape(2,2,2)
	# q = np.array([1,4,3,5,6,7])
	# print q
	# print list(q[[1,2,3]])
	# print kernel2D_connections

	# print test.return_flat()
	# print test.get_1U_neighbor()
	# print test.get_2U_neighbor()
	# print test.get_3U_neighbor()
	# print test.get_POI()




	# q = [i for i in neighbors if i != 0]

	# print layer_comparator(stack)
	# print a
	# print b
	# a = [(2,2),(1,2),(9,9),(5,3)]
	# b = [(1,2),(2,1), (2,3)]
	# print "================"
	# for r in b:
	# 	print r in a or tuple(reversed(r)) in a

	# print [a[item]==0 for item in a]
	# q = all(v == 0 for v in row for row in a)

	# print any(0 in sublist for sublist in q)
