import sys
from lib.render import *
from lib.processing import *
import lib.pathfinder as pathfinder
from lib.read_write import *
from lib.math_funcs import *

import time
import numpy as np
import scipy.io
import argparse
from scipy import ndimage as ndi
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
	graph_map = pathfinder.Graph()
	# Create an array of IDs
	item_id = np.array(range(0, zdim * xdim * ydim)).reshape(zdim, xdim, ydim)
	# Traverse input binary image
	# print "\tSlices Analyzed: ",
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
	# for z in xrange(zdim):
	# 	for x in xrange(xdim):
	# 		for y in xrange(ydim):
	# 			# Get Query ID Node #
	# 			query_ID = item_id[z, x, y]
	# 			# Get neighbors to Query
	# 			neighbor_locations = get_3d_neighbor_coords((z, x, y), input_binary.shape)
	# 			# For each neighbor
	# 			for neighbor in neighbor_locations:
	# 				# Get Neighbor ID
	# 				neighbor_ID = item_id[neighbor]
	# 				# If query exists and neighbor exists, branch query and neighbor.
	# 				# If only Query exists, branch query to itself.
	#
	# 				if input_binary[z, x, y]:
	# 					if input_binary[neighbor]:
	# 						graph_map.addEdge(origin = query_ID,
	# 											destination = neighbor_ID,
	# 											bidirectional = False,
	# 											self_connect = True)
	# 					else:
	# 						graph_map.addEdge(origin = query_ID,
	# 											destination = query_ID,
	# 											bidirectional = False,
	# 											self_connect = True)
	# 				else:
	# 					pass
	# 	print z,
	# print "\n"
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
	for key in graph_dict.keys():
		try:
			network = sorted(graph.BFS(key))
			for connected_key in network:
				graph_dict.pop(connected_key, None)
			if network not in network_element_list:
				network_element_list.append(network)
		except:
			pass
	print "> Unique Paths + Background [1]: ", len(network_element_list)

	img_dimensions = ID_map.shape
	output = np.zeros_like(ID_map).flatten()

	last_used_label = 1
	print "> Labeling Network"
	for network in network_element_list:
		for element in network:
			output[element] = last_used_label
		last_used_label += 1
	return output.reshape(img_dimensions)


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


def get_attributes(masked_image, x = 1.0, y = 1.0, stack_height = 1.0):
	masked_image[masked_image > 0] = 1
	volume = np.sum(masked_image) * stack_height

	masked_image = masked_image.astype(bool)
	# print "> Computing surface..."

	verts, faces, normals, values = measure.marching_cubes_lewiner(masked_image,
																	level = None,
																	spacing = (x, y, stack_height),
																	gradient_direction = 'descent',
																	step_size = 1,
																	allow_degenerate = True,
																	use_classic = False)
	triangle_collection = verts[faces]
	# print "> Computing attributes..."
	triangle_Surface = Surface(triangle_collection)
	nTriangles, surfaceArea = triangle_Surface.get_stats()
	return volume, nTriangles, surfaceArea


def reverse_cantor_pair(z):
	w = np.floor((np.sqrt((8 * z) + 1) - 1) / 2)
	t = (w ** 2 + w) / 2
	return int(w - z + t), int(z - t)


def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for 3d segmenting mitochondria')
	parser.add_argument('-r',
						dest = 'results_dir',
						help = 'Main results directory',
						required = False,
						default = ".\\test_run")

	options = vars(parser.parse_args())
	return options


def main(save_dir):
	try:
		options = get_args(sys.argv)
		save_dir = options['results_dir']
		print "> ==========================================================================================\n"
		print "> Data Directory: {}\n".format(save_dir)
	except:
		sys.exit()
	print "> ==========================================================================================\n"
	print "> Starting 3D Segmentation and characterizing module\n"

	save_dir_anal = os.path.join(save_dir, 'analysis')
	cell_mito_data_pairs = read_txt_file(os.path.join(save_dir_anal, "Cell_mito_UUID_Pairs.txt"))

	save_dir_3D = os.path.join(save_dir, '3D_seg')
	mkdir_check(save_dir_3D)

	mito_stats = open(os.path.join(save_dir, "mitochondria_statistics.txt"), "w")
	processing_stats = open(os.path.join(save_dir, "3DS_processing_time.txt"), "w")
	for Cell_UUID, Mito_UUID, cell_FID, mito_FID, origin_path, _ in cell_mito_data_pairs:
		start = time.time()
		print "> ==========================================================================================\n"
		print "> Query C: {}".format(Cell_UUID)
		print "> Query M: {}\n".format(Mito_UUID)
		CM_filename = "CM_" + Cell_UUID + "_bin.mat"
		CMS_filename = "CSM_" + Cell_UUID + "_3DS.mat"

		cell_mitos = scipy.io.loadmat(os.path.join(save_dir_anal, CM_filename))['data']

		post_segmentation = layer_comparator(cell_mitos)

		labeled = stack_cantor_multiplier(cell_mitos, post_segmentation)

		prior_seg_element_num = len(set(cell_mitos.flatten()))
		post_3d_element_num = len(set(labeled.flatten())) # includes background

		print "> Pre: {}\tPost: {}\t segmentation bodies".format(prior_seg_element_num, post_3d_element_num)

		for element_num in set(labeled.flatten()):
			cell_num, mito_num = reverse_cantor_pair(element_num)
			if cell_num == 0 or mito_num == 0:
				pass
			else:
				mask = np.zeros_like(labeled)
				mask[labeled == element_num] = element_num
				vol, nTriangle, SA = get_attributes(mask, x = 1.0, y = 1.0, stack_height = 3.458)
				mito_stats.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(Cell_UUID, Mito_UUID, cell_FID, mito_FID, origin_path, cell_num, mito_num, vol, nTriangle, SA))

		save_data(labeled, CMS_filename, save_dir_3D)
		end = time.time()
		processing_stats.write(str(end-start) + "\n")
	print "> Mitochondria Statistics written to {}".format(os.path.join(save_dir, "mitochondria_statistics.txt"))
	mito_stats.close()
	processing_stats.close()
if __name__ == "__main__":
	main(sys.argv)

	# # stack_viewer(stack)
	# labeled = layer_comparator(stack)
	# stack_viewer(labeled)
	# print "===========>"
	# image = scipy.io.loadmat("C:\\Users\\Gordon Sun\\Documents\\GitHub\\bootlegged_pipeline\\test_run\\analysis\\CM_5f1701bd4f534712925c9b0739503215_bin.mat")['data']
	# n_element = len(set(image.flatten()))
	# print set(image.flatten())
	# print n_element
	# test = [tuple(point) for point in np.argwhere(image == 1)]
	# print len(np.argwhere(image == 1))
	# print image.shape
	# post = layer_comparator(image)
	# stack_viewer(post)
	# print "====="


	# my_dict = { 0 : [2,3,4],
	# 			1 : [3,45,5],
	# 			2 : [3,4,5],
	# 			4 : [6,4,5],
	# 			5 : [1,2,3],
	# 			6 : [5,5,6,6],
	# 			9 : [2,3,4]
	# 			}
	# print my_dict
	# for key in my_dict.keys():
	# 	try:
	# 		for element in my_dict[key]:
	# 			my_dict.pop(element, None)
	# 	except:
	# 		pass
	# print my_dict
	# print test
	# for coord in test:
	# 	print image[coord]

	# image[image != 1] = 0
	# stack_viewer(image)
	# layer_comparator(image)
	# stack_viewer(image)
	# print get_attributes(image, stack_height = 0.5)
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
