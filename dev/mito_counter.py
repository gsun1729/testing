from lib.render import *
import dev.pathfinder
import numpy as np
import scipy.io
import sys
import argparse
from lib.point import *
from scipy import ndimage as ndi
from skimage import measure
from lib.math_funcs import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib.read_write import *


class rect_prism(object):
	def __init__(self, matrix):
		self.dimension_tuple = matrix.shape


	def corner_locations(self):
		'''
		Given n dimensions, returns the coordinates of where the corners should be in the given space in the form of a list of lists

		:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
		:return: list of lists with sublists containing coordinates of the corner space
		'''
		corners = []
		for x in xrange(2 ** len(self.dimension_tuple)):
			binary = str(np.binary_repr(x))
			if len(binary) < len(self.dimension_tuple):
				binary = str(0) * (len(self.dimension_tuple) - len(binary)) + binary
			empty_corner = list(np.zeros(len(binary), dtype = int))
			for coord in xrange(len(binary)):
				empty_corner[coord] = int(binary[coord], 2) * int(self.dimension_tuple[coord] - 1)
			corners.append(tuple(empty_corner))
		return corners


	def edge_locations(self):
		'''
		Hardcoded edge detection
		num edge elements in an ndimensional element = (4 * (np.sum(dimension_tuple) - (2 * len(dimension_tuple)))):
		Can only interpret edges in a 2d or 3d volume.

		:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
		:return: list of lists with sublists containing coordinates of the edges
		'''
		edges = []
		if len(self.dimension_tuple) == 2:
			x_dim, y_dim = self.dimension_tuple
			for x in xrange(x_dim):
				for y in xrange(y_dim):
					if (x == 0 or x == x_dim - 1) or (y == 0 or y == y_dim - 1):
						edges.append((x, y))

		elif len(self.dimension_tuple) == 3:
			z_dim, x_dim, y_dim = self.dimension_tuple
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
		corners = self.corner_locations()
		edges = [edge for edge in edges if edge not in corners]
		return edges


	def face_locations(self):
		'''
		Hardcoded face detection
		Can only interpret faces in a 2d or 3d volume.

		:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
		:return: list of lists with sublists containing coordinates of the faces
		'''
		faces = []
		if len(self.dimension_tuple) == 2:
			print "Faces don't exist for 2D geometries"
			return faces
		elif len(self.dimension_tuple) == 3:
			z_dim, x_dim, y_dim = self.dimension_tuple
			for z in xrange(z_dim):
				for x in xrange(x_dim):
					for y in xrange(y_dim):
						if (z == 0 or z == z_dim - 1):
							faces.append((z, x, y))
						if (x == 0 or x == x_dim - 1):
							faces.append((z, x, y))
						if (y == 0 or y == y_dim - 1):
							faces.append((z, x, y))
		corners = self.corner_locations()
		edges = self.edge_locations()
		faces = [face for face in faces if face not in corners and face not in edges]
		return faces


	def core_locations(self):
		'''
		Hardcoded core detection
		Can only interpret faces in a 2d or 3d volume.

		:param dimension_tuple: a tuple listing the length of the dimensions of the space in question
		:return: list of lists with sublists containing coordinates of the cores
		'''
		cores = []
		if len(self.dimension_tuple) == 2:
			x_dim, y_dim = self.dimension_tuple
			for x in xrange(x_dim):
				for y in xrange(y_dim):
					cores.append((x, y))

		elif len(self.dimension_tuple) == 3:
			z_dim, x_dim, y_dim = self.dimension_tuple
			for z in xrange(z_dim):
				for x in xrange(x_dim):
					for y in xrange(y_dim):
						cores.append((z, x, y))
		corners = self.corner_locations()
		edges = self.edge_locations()
		faces = self.face_locations()
		cores = [core for core in cores if core not in corners and core not in edges and core not in faces]
		return cores


	def is_core(self, query):
		if query not in self.core_locations():
			return False
		else:
			return True


	def is_corner(self, query):
		if query not in self.corner_locations():
			return False
		else:
			return True


	def is_face(self, query):
		if query not in self.face_locations():
			return False
		else:
			return True


	def is_edge(self, query):
		if query not in self.edge_locations():
			return False
		else:
			return True


def get_3d_neighbor_coords(tuple_location, size):
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


def imglattice2graph(input_binary):
	zdim, xdim, ydim = input_binary.shape
	# Instantiate graph
	graph_map = dev.pathfinder.Graph()
	# Create an array of IDs
	item_id = np.array(range(0, zdim * xdim * ydim)).reshape(zdim, xdim, ydim)
	# Traverse input binary image
	print "\tSlices Analyzed: ",
	for z in xrange(zdim):
		for x in xrange(xdim):
			for y in xrange(ydim):
				# Get Query ID Node #
				query_ID = item_id[z, x, y]
				# Get neighbors to Query
				neighbor_locations = get_3d_neighbor_coords((z, x, y), input_binary.shape)
				# For each neighbor
				for neighbor in neighbor_locations:
					# Get Neighbor ID
					neighbor_ID = item_id[neighbor]
					# If query exists and neighbor exists, branch query and neighbor.
					# If only Query exists, branch query to itself.

					if input_binary[z, x, y]:
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
		print z,
	print "\n"
	return item_id, graph_map


def layer_comparator(image3D):
	print "> Generating lattice"
	ID_map, graph = imglattice2graph(image3D)
	graph_dict = graph.get_self()
	# for key in sorted(graph_dict.iterkeys()):
	# 	print "%s: %s" % (key, graph_dict[key])
	network_element_list = []
	print "> Network size: ", len(graph_dict)
	# print graph_dict
	print "> Pruning Redundancies"
	for key in graph_dict.iterkeys():
		network = sorted(graph.BFS(key))
		# print key,
		if network not in network_element_list:
			network_element_list.append(network)
	print "> Unique Paths: ", len(network_element_list)
	last_used_label = 1
	print "> Labeling Network"
	for network in network_element_list:
		for element in network:
			image3D[np.where(ID_map == element)] = last_used_label
		last_used_label += 1
	return image3D


def euclid_dist_nD(p0, p1):
	return np.sum((p1 - p0) ** 2) ** 0.5


class Point_set(object):
	def __init__(self, point_list):
		self.point_list = np.array([[float(coordinate) for coordinate in point] for point in point_list])
		self.num_pts = len(self.point_list)


	def perimeter(self):
		peri_distance = 0
		for pt_indx in xrange(self.num_pts):
			peri_distance += euclid_dist_nD(self.point_list[pt_indx],
											self.point_list[pt_indx - 1])
		return peri_distance


	def side_lengths(self):
		side_len = []
		for pt_indx in xrange(self.num_pts):
			side_len.append(euclid_dist_nD(self.point_list[pt_indx],
											self.point_list[pt_indx - 1]))
		return np.array(side_len)


	def heron_area(self):
		semi_peri = self.perimeter() / 2
		prod = semi_peri
		for side in self.side_lengths():
			prod *= semi_peri - side
		return np.sqrt(prod)


class Surface(object):
	def __init__(self, triangle_collection):
		self.triangle_collection = triangle_collection
		self.num_triangles = len(triangle_collection)
		self.SA = self.get_SA()


	def get_SA(self):
		total = 0
		for triangle in self.triangle_collection:
			triangle_set = Point_set(triangle)
			total += triangle_set.heron_area()
		return total


	def get_stats(self):
		return self.num_triangles, self.SA


def get_attributes(masked_image, stack_height = 1.0):
	masked_image[masked_image > 0] = 1
	volume = np.sum(masked_image) * stack_height

	masked_image = masked_image.astype(bool)
	print "> Computing surface..."
	verts, faces, normals, values = measure.marching_cubes_lewiner(masked_image,
																	level = None,
																	spacing = (stack_height, 1.0, 1.0),
																	gradient_direction = 'descent',
																	step_size = 1,
																	allow_degenerate = True,
																	use_classic = False)
	triangle_collection = verts[faces]
	print "> Computing Surface Area..."
	triangle_Surface = Surface(triangle_collection)
	nTriangles, surfaceArea = triangle_Surface.get_stats()
	return volume, nTriangles, surfaceArea


def main(read_path):
	image = scipy.io.loadmat(read_path)['data']

	z, x, y = image.shape
	image[image > 0] = 1
	stack_viewer(image)


if __name__ == "__main__":
	# stack = np.zeros((30,30,30))
	# stack[10:20,10:20,10:20] = 1
	# labeled = layer_comparator(stack)
	# print stack
	# print labeled
	# # print type(labeled)
	# # print type(labeled[0,0,0])
	image = scipy.io.loadmat("C:\\Users\\Gordon Sun\\Documents\\GitHub\\bootlegged_pipeline\\test_run\\analysis\\CM_1ca89836c92b49ae8aa2d76515f25bf6_bin.mat")['data']
	print set(image.flatten())
	image[image != 8] = 0
	# stack_viewer(image)
	# layer_comparator(image)
	# stack_viewer(image)
	print get_attributes(image, stack_height = 0.5)
	# print np.sum(image)
	#
	# verts = np.array([[1,2,3], [2,2,2], [3,3,3], [1,2,2], [4,5,3], [5,5,5]])
	# faces = np.array([[1,2,3], [1,2,4], [0,2,3]])
	# print verts
	# print faces
	# triangles =  verts[faces]
	# print "===="
	# print triangles
	# print "==="
	# for triangle in triangles:
	# 	print triangle
	#
	# print "======="
	# WOW = Surface(triangles)
	# print WOW.get_stats()
	# print WOW.ge
	# labeled[labeled !=3] = 0
	# print labeled

	# get_attributes(image)
