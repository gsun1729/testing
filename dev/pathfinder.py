# Program to print BFS traversal from a given source
# vertex. BFS(int s) traverses vertices reachable
# from s.
# https://www.geeksforgeeks.org/?p=18382
from collections import defaultdict
import sys
import numpy as np
# This class represents a directed graph using adjacency
# list representation

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
	def __init__(self):
		# default dictionary to store graph
		self.graph = defaultdict(list)


	def addEdge(self, origin, destination, bidirectional = False, self_connect = True):
		'''
		Function to add an edge to graph, can be set to bidirectional if desired
		Manual entry of each element
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


	def connections2graph(self, connection_table, connection_direction, *exist_list):
		'''
		Function creates a bidirectional graph given a 2d table of connections between points
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
		'''
		Function to print a BFS(Breadth First Traversal) of graph
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
		'''
		Given a start point and an end point, determine whether if the two points are connected by any path.
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
		return self.graph



# Driver code
# Create a graph given in the above diagram
g = Graph()
#
g.connections2graph(paths, path_direction, np.array([0,0,1,1,0,1,1,1]))
print g.get_self()
# print g.BFS(5)/
#
print g.path_exists(2,12)
print g.path_exists(1,2)
print g.path_exists(2,2)
print g.path_exists(3,2)
print g.path_exists(4,2)
print g.path_exists(5,2)
print g.path_exists(6,2)
print g.path_exists(7,2)
print g.path_exists(8,2)
print g.path_exists(8,8)
