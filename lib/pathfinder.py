from collections import defaultdict
import sys
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

'''Module contains graph generation functions and classes intended for 3d image segmentation and characterization
'''
# This matrix represents a directed graph using adjacency
# list representation for a 8 vertex cube
paths = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
				 [1, 1, 0, 1, 0, 1, 0, 0],
				 [1, 0, 1, 1, 0, 0, 1, 0],
				 [0, 1, 1, 1, 0, 0, 0, 1],
				 [1, 0, 0, 0, 1, 1, 1, 0],
				 [0, 1, 0, 0, 1, 1, 0, 1],
				 [0, 0, 1, 0, 1, 0, 1, 1],
				 [0, 0, 0, 1, 0, 1, 1, 1]])
path_direction = np.zeros((8, 8), dtype = bool)

class Graph:
	'''Class for creating graphs for 3d image segmentation'''
	def __init__(self):
		# default dictionary to store graph
		self.graph = defaultdict(list)


	def addEdge(self, origin, destination, bidirectional = False, self_connect = True):
		'''Function to add an edge to graph, can be set to bidirectional if desired
		Manual entry of each element

		:param origin: [int] start node ID
		:param destination: [int] end node ID
		:param bidirectional: [bool] bool indicating whether the connection is bidirectional
		:param self_connect: [bool] indicate whether the origin node connects to itself.
		'''
		# Append edge to dictionary of for point
		self.graph[origin].append(destination)
		# Append origin node edge to itself
		if self_connect:
			self.graph[origin].append(origin)
		# Append node edge to itself
		self.graph[destination].append(destination)
		# Append reverse direction if bidirectional
		if bidirectional:
			self.graph[destination].append(origin)
		# Remove duplicates
		self.graph[origin] = list(set(self.graph[origin]))
		self.graph[destination] = list(set(self.graph[destination]))


	def rmEdge(self, origin, destination):
		'''Function tries to delete an edge in a graph, conditional on if it exists

		:param origin: [int] origin node number
		:param destination: [int] Destination node number
		'''

		if self.path_exists(origin, destination):
			origin_connections = len(self.graph[origin])
			dest_connections = len(self.graph[destination])
			self.graph[origin].remove(destination)
			if origin == destination:
				pass
			else:

				if origin_connections == 1 and dest_connections == 1:
					pass
				else:
					self.graph[destination].remove(origin)
		else:
			raise Exception("Path from {} to {} does not exist".format(origin, destination))


	def rmNode(self, node):
		'''Function removes a node and all of its associated connections from the graph

		:param node: [int] node to be removed
		'''
		connections = copy.deepcopy(self.graph[node])
		for connection in connections:
			self.graph[connection].remove(node)
		self.graph.pop(node, None)


	def connections2graph(self, connection_table, connection_direction, *exist_list):
		'''Function creates a bidirectional graph given a 2d table of connections between points

		:param connection_table: [nd.array] numpy binary adjacency matrix
		:param connection_direction: [np.ndarray] numpy matrix of m x n bools
		:param exist_list: [list] list of whether elements within the axes of the adjacency matrix exist
		'''
		if not exist_list:
			exist_list = np.ones(len(connection_table))
		else:
			exist_list = exist_list[0]

		x_dim, y_dim = connection_table.shape
		exists = np.outer(exist_list, exist_list.T)
		connection_table = exists * connection_table
		# print connection_table
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if connection_table[x, y] == 1:
					self.addEdge(x, y, bidirectional = connection_direction[x, y])
				else:
					pass


	def BFS(self, s):
		'''Function to print a BFS(Breadth First Traversal) of graph

		:param s: [int] query node ID number
		'''
		connections = []
		# If element is not even in graph, there is no way to start from it
		if not s in self.graph:
			return connections
		# Mark all the vertices as not visited
		visited = [False]*(len(self.graph))
		dict_visted = dict(zip(self.graph.keys(), visited))
		# print dict_visted
		# Create a queue for BFS
		queue = []

		# Mark the source node as visited and enqueue it
		queue.append(s)
		dict_visted[s] = True
		# # print queue
		while queue:
		# 	# Dequeue a vertex from queue and print it
			s = queue.pop(0)
			# print s,
			connections.append(s)
			# Get all adjacent vertices of the dequeued
			# vertex s. If a adjacent has not been visited,
			# then mark it visited and enqueue it
			for i in self.graph[s]:
				if dict_visted[i] == False:
					queue.append(i)
					dict_visted[i] = True
		return connections


	def path_exists(self, start, end):
		'''Given a start point and an end point, determine whether if the two points are connected by any path.

		:param start: [int] node ID for starting node
		:param end: [int] node ID for ending node
		'''
		if not start in self.graph or not end in self.graph:
			return False
		else:
			if start == end:
				return True
			else:
				connections = self.BFS(start)
				if any(v == end for v in connections):
					return True
				else:
					return False


	def get_self(self):
		'''Statement used for getting graph contents for printing and debugging
		'''
		return self.graph


	def get_cliques(self):
		'''Using networkx find cliques algorithm (Bron Kerbosch)
		'''
		nx_graph = nx.from_dict_of_lists(self.graph)
		max_cliques = nx.find_cliques(nx_graph)
		return max_cliques


	def rm_max_cliques(self):
		'''
		scans through max cliques and replaces them with a singular node if the max clique size is larger than 2

		'''
		max_clique_ls = self.get_cliques()
		max_key = np.amax(list(self.graph.keys()))
		last_nnode = max_key
		for clique in max_clique_ls:
			if len(clique) <= 2: 
				continue
			else:
				new_node = last_nnode + 1
				last_nnode = new_node
				for node in clique:
					for connection in self.graph[node]:
						self.addEdge(new_node, connection, 
											bidirectional = True, 
											self_connect = True)
					self.rmNode(node)


	def check_empty(self):
		'''Hacky fix for dealing with single px images with no neighbors, result is that node ends up getting recorded as nonexistent
		:return: False if graph has elements, True if it is Empty
		'''

		if not self.graph.keys():
			self.graph[1] = [1]
			return True
		else:
			return False


	def num_junctions(self):
		'''count the number of junctions in a graph and return marked junctions w/ connections
		'''
		junctions = {}
		for key, values in self.graph.items():
			temp_connections = copy.deepcopy(values)
			temp_connections.remove(key)
			if len(temp_connections) > 2:
				junctions[key] = temp_connections
			else:
				pass
		return junctions


	def num_endpoints(self):
		'''count the number of junctions in a graph and return marked junctions w/ connections
		'''
		endpts = {}
		for key, values in self.graph.items():
			temp_connections = copy.deepcopy(values)
			temp_connections.remove(key)
			if len(temp_connections) < 2:
				endpts[key] = temp_connections
			else:
				pass
		return endpts


class Neighbors_3D:
	''' Given a point and defined boundaries (if desired), determine neighbors based on manhattan distance
	'''
	def __init__(self, search_origin, boundaries, define_boundaries = True):
		'''
		stores search origin coordinates and limits for searching for neighbors based on image size
		:param search_origin: [list] 3 item list indicating the 3 z,x,y coordinates for the query
		:param boundaries: [list] 3 item list indicating the maximum dimensions in the search space
		:param define_boundaries: [bool] param whether or not to restrict neighbors to a specified search space
		'''
		self.z = search_origin[0]
		self.x = search_origin[1]
		self.y = search_origin[2]
		self.defineB = define_boundaries
		self.zlim = boundaries[0]
		self.xlim = boundaries[1]
		self.ylim = boundaries[2]


	def set_limits(self, neighbor_list):
		'''
		filters out for points that fall outside of the search range given the search space
		:param neighbor_list: [list] of [lists] containing cartesian coordinates for the points in neighbors
		:return neighbor_list: [list] of [lists] similar to that of input
		'''
		if self.defineB:
			return [pt for pt in neighbor_list if (pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0) and (pt[0] < self.zlim and pt[1] < self.xlim and pt[2] < self.ylim)]
		else:
			return neighbor_list


	def adjacent_1U(self):
		'''
		retrieves neighbors within 1 manhattan distance unit away from the search origin
		'''
		top = (self.z + 1, self.x, self.y)
		bottom = (self.z - 1, self.x, self.y)
		front = (self.z, self.x + 1, self.y)
		back = (self.z, self.x - 1, self.y)
		left = (self.z, self.x, self.y - 1)
		right = (self.z, self.x, self.y + 1)

		neighbors = [top, bottom, front, back, left, right]
		neighbors = self.set_limits(neighbors)
		return neighbors


	def adjacent_2U(self):
		'''
		retrieves neighbors within 2 manhattan distance unit away from the search origin
		'''
		edge1 = (self.z - 1, self.x, self.y - 1)
		edge2 = (self.z - 1, self.x + 1, self.y)
		edge3 = (self.z - 1, self.x, self.y + 1)
		edge4 = (self.z - 1, self.x - 1, self.y)
		edge5 = (self.z, self.x - 1, self.y - 1)
		edge6 = (self.z, self.x + 1, self.y - 1)
		edge7 = (self.z, self.x + 1, self.y + 1)
		edge8 = (self.z, self.x - 1, self.y + 1)
		edge9 = (self.z + 1, self.x, self.y - 1)
		edge10 = (self.z + 1, self.x + 1, self.y)
		edge11 = (self.z + 1, self.x, self.y + 1)
		edge12 = (self.z + 1, self.x - 1, self.y)

		neighbors = [edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12]
		neighbors = self.set_limits(neighbors)
		return neighbors


	def adjacent_3U(self):
		'''
		retrieves neighbors within 3 manhattan distance unit away from the search origin
		'''
		corner1 = (self.z - 1, self.x - 1, self.y - 1)
		corner2 = (self.z - 1, self.x + 1, self.y - 1)
		corner3 = (self.z - 1, self.x + 1, self.y + 1)
		corner4 = (self.z - 1, self.x - 1, self.y + 1)
		corner5 = (self.z + 1, self.x - 1, self.y - 1)
		corner6 = (self.z + 1, self.x + 1, self.y - 1)
		corner7 = (self.z + 1, self.x + 1, self.y + 1)
		corner8 = (self.z + 1, self.x - 1, self.y + 1)
		
		neighbors = [corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8]
		neighbors = self.set_limits(neighbors)
		return neighbors


	def adjacent_range(self, radius_range):
		'''
		retrieves neighbors in a defined radius range (defined by manhattan distance)
		:param radius_range: [list] of [str] with high and low distances, with string values being 1U, 2U, or 3U
		'''
		llim, hlim = radius_range
		neighbors = []
		if llim == '1U':
			neighbors.extend(self.adjacent_1U())
			if hlim == '2U':
				neighbors.extend(self.adjacent_2U())
			elif hlim == '3U':
				neighbors.extend(self.adjacent_2U())
				neighbors.extend(self.adjacent_3U())
			else:
				raise Exception('hlim value not recognized in radius {}'.format(radius_range))
		elif llim == '2U':
			neighbors.extend(self.adjacent_2U())
			if hlim == '3U':
				neighbors.extend(self.adjacent_3U())
			elif hlim == '1U':
				neighbors.extend(self.adjacent_1U())
			else:
				raise Exception('hlim value not recognized in radius {}'.format(radius_range))
		elif llim == '3U':
			if hlim == '2U':
				neighbors.extend(self.adjacent_2U())
				neighbors.extend(self.adjacent_3U())
			elif hlim == '1U':
				neighbors.extend(self.adjacent_1U())
				neighbors.extend(self.adjacent_3U())
			else:
				raise Exception("hlim value not recognized in radius {}".format(radius_range))
		else:
			raise Exception("adjacent range works within 1-3U of origin query, and must accept a range of 1-3, use single adjacency functions for discrete distances")
		return neighbors

	def get_all(self):
		'''
		gets all neighbors within 3U radius
		'''
		neighbors = []
		neighbors.append(self.adjacent_1U)
		neighbors.append(self.adjacent_2U)
		neighbors.append(self.adjacent_3U)
		return neighbors


class rect_prism(object):
	'''
	Given a rectangular prism, defines different classes on the rectangular prism surfaces
	call8ing different functions within class provides coordinates for all points on the 
	selected structure of the geometry.
	'''
	def __init__(self, matrix):
		self.dimension_tuple = matrix.shape


	def corner_locations(self):
		'''Given n dimensions, returns the coordinates of where the corners should be in the given space in the form of a list of lists

		:param dimension_tuple: [tuple] a tuple listing the length of the dimensions of the space in question
		:return: [list] of [list] list of lists with sublists containing coordinates of the corner space
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
		'''Hardcoded edge detection
		num edge elements in an ndimensional element = (4 * (np.sum(dimension_tuple) - (2 * len(dimension_tuple)))):
		Can only interpret edges in a 2d or 3d volume.

		:param dimension_tuple: [tuple] a tuple listing the length of the dimensions of the space in question
		:return: [list] of [list] list of lists with sublists containing coordinates of the edges
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
		'''Hardcoded face detection
		Can only interpret faces in a 2d or 3d volume.

		:param dimension_tuple: [tuple] a tuple listing the length of the dimensions of the space in question (3D)
		:return: [list] of [list] list of lists with sublists containing coordinates of the faces
		'''
		faces = []
		if len(self.dimension_tuple) == 2:
			print("Faces don't exist for 2D geometries")
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
		'''Hardcoded core detection
		Can only interpret faces in a 2d or 3d volume.

		:param dimension_tuple: [tuple] a tuple listing the length of the dimensions of the space in question
		:return: [list] of [list] list of lists with sublists containing coordinates of the cores
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
		'''returns bool if the location of the index is a core location

		:param query: [tuple] location of the query in tuple form (3d)
		:return: [bool]
		'''
		if query not in self.core_locations():
			return False
		else:
			return True


	def is_corner(self, query):
		'''returns bool if the location of the index is a corner location

		:param query: [tuple] location of the query in tuple form (3d)
		:return: [bool]
		'''
		if query not in self.corner_locations():
			return False
		else:
			return True


	def is_face(self, query):
		'''returns bool if the location of the index is a face location

		:param query: [tuple] location of the query in tuple form (3d)
		:return: [bool]
		'''
		if query not in self.face_locations():
			return False
		else:
			return True


	def is_edge(self, query):
		'''returns bool if the location of the index is a edge location

		:param query: [tuple] location of the query in tuple form (3d)
		:return: [bool]
		'''
		if query not in self.edge_locations():
			return False
		else:
			return True


def imglattice2graph(input_binary, neighbor_distance = '3U'):
	'''Converts a 3d image into a graph for segmentation

	:param input_binary: [np.ndarray] complete binary image 3d
	:param neighbor_distance: [str] or [list] defined distance for neighbors
	:return item_id: [np.ndarray] indicies of all elements in the lattice for identification
	:return graph_map: [graph object] graph object indicating which voxels are connected to which voxels
	'''
	zdim, xdim, ydim = input_binary.shape
	# Instantiate graph
	graph_map = Graph()
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
				neighbors = Neighbors_3D(location, input_binary.shape)
				if isinstance(neighbor_distance, list):
					neighbor_locations = neighbors.adjacent_range(neighbor_distance)
				elif isinstance(neighbor_distance, str):
					if neighbor_distance is '3U':
						neighbor_locations = neighbors.adjacent_3U()
					elif neighbor_distance is '2U':
						neighbor_locations = neighbors.adjacent_2U()
					elif neighbor_distance is '1U':
						neighbor_locations = neighbors.adjacent_1U()
					else:
						raise ValueError("Defined Neighbor distance radius not recognized: {}".format(neighbor_distance))
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


def layer_comparator(image3D):
	'''
	Uses lattice graph data to determine where the unique elements are and prune redundancies.
	Turns individual pixel connections into a dictionary of individual elements labeled with [key]
	[values] indicate pixel IDs associated with structure in [key]
	:param image3D: [np.ndarray] original binary image 3d
	:return: [np.ndarray] segmented 3d image
	'''
	print("> Generating lattice")
	ID_map, graph = imglattice2graph(image3D)

	graph_dict = graph.get_self()
	# for key in sorted(graph_dict.iterkeys()):
	# 	print("%s: %s" % (key, graph_dict[key]))
	network_element_list = []
	print("> Network size: ", len(graph_dict))
	# print(graph_dict)
	print("> Pruning Redundancies")
	for key in graph_dict.keys():
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


def prune_graph(graph_object):
	if graph_object.check_empty():
		graph_object.check_empty()
	else:
		graph_object.rm_max_cliques()

	return graph_object


def filter_unitcycles(list_cycles):
	long_cycles = []
	for cycle in list_cycles:
		if len(cycle) == 1:
			pass
		else:
			long_cycles.append(cycle)
	if long_cycles:
		return long_cycles
	else:
		return None



def process_graph(graph_object):
	graph = graph_object.get_self()
	reduced_graph = prune_graph(graph_object)
	junct = graph_object.num_junctions()
	endpt = graph_object.num_endpoints()

	# convert graph to dict for transfer to networkx handling
	reduced_graph = dict(reduced_graph.get_self())
	nx_graph = nx.from_dict_of_lists(reduced_graph)
	n_edges = np.ceil((nx_graph.number_of_edges() - 1) / 2)

	cycles = nx.cycle_basis(nx_graph)

	ecc = nx.eccentricity(nx_graph)
	max_ecc = max(ecc.values())
	min_ecc = min(ecc.values())
	brdge = list(nx.bridges(nx_graph))
	print(len(brdge))
	print(n_edges, len(endpt), len(junct), filter_unitcycles(cycles))
	print(min_ecc, max_ecc)
	# print(ecc)
	# A = nx.nx_agraph.to_agraph(nx_graph)        
	# A.layout(prog='dot')
	# A.draw('file.png')
	nx.draw_networkx(nx_graph)
	plt.show()


if __name__ == "__main__":
	# Create a graph given in the above path listing
	g = Graph()
	g.connections2graph(paths, path_direction, np.array([0,0,1,1,0,1,1,1]))
	print(g.get_self())
	print(g.get_self()[2])
	g.rmEdge(3,0)
	print(g.get_self())
	# print g.BFS(5)/
	print(g.path_exists(2,12))
	print(g.path_exists(1,2))
	print(g.path_exists(2,2))
	print(g.path_exists(3,2))
	print(g.path_exists(4,2))
	print(g.path_exists(5,2))
	print(g.path_exists(6,2))
	print(g.path_exists(7,2))
	print(g.path_exists(8,2))
	print(g.path_exists(8,8))
