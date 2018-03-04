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


	def addEdge(self, origin, destination, bidirectional = False):
		'''
		Function to add an edge to graph, can be set to bidirectional if desired
		Manual entry of each element
		'''
		# Append edge to dictionary of for point
		self.graph[origin].append(destination)
		# Append origin node edge to itself
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
		# print exists * connection_table
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if connection_table[x, y] == 1 and exists[x, y] == 1:
					self.addEdge(x, y, bidirectional = connection_direction[x, y])
				else:
					pass


	def BFS(self, s):
		'''
		Function to print a BFS(Breadth First Traversal) of graph
		'''
		# Mark all the vertices as not visited
		visited = [False]*(len(self.graph))
		# Create a queue for BFS
		queue = []
		connections = []
		# Mark the source node as visited and enqueue it
		queue.append(s)
		visited[s] = True
		# print queue
		while queue:
			# Dequeue a vertex from queue and print it
			s = queue.pop(0)
			# print s,
			connections.append(s)
			# Get all adjacent vertices of the dequeued
			# vertex s. If a adjacent has not been visited,
			# then mark it visited and enqueue it
			for i in self.graph[s]:
				if visited[i] == False:
					queue.append(i)
					visited[i] = True
		return connections


	def path_exists(self, start, end):
		'''
		Given a start point and an end point, determine whether if the two points are connected by any path.
		'''
		connections = self.BFS(start)
		if any(connection == end for connection in connections):
			return True
		else:
			return False


	def get_self(self):
		print self.graph



# Driver code
# Create a graph given in the above diagram
g = Graph()
print paths

g.connections2graph(paths, path_direction, np.array([0,0,1,0,0,1,1,1]))
print g.get_self()
# print np.outer(np.array([0,0,1,0,0,1,1,1]), np.array([0,0,1,0,0,1,1,1]).T)
# g.get_self()
# print g.BFS(2)
# print g.path_exists(2, 5)

sys.exit()
print "Following is Breadth First Traversal (starting from vertex 2)"
g.BFS(0)
print "\n"
g.BFS(1)
print "\n"
g.BFS(2)
print "\n"
print g.BFS(0)


# This code is contributed by Neelam Yadav
