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

	# Function to add an edge to graph
	def addEdge(self, origin, destination, bidirectional = False):
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

	# Function creates a bidirectional graph given a 2d table of connections between points
	def connections2graph(self, connection_table, connection_direction):
		x_dim, y_dim = connection_table.shape
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if connection_table[x, y] == 1:
					self.addEdge(x, y, bidirectional = connection_direction[x, y])
				else:
					pass

	# Function to print a BFS of graph
	def BFS(self, s):
		# Mark all the vertices as not visited
		visited = [False]*(len(self.graph))
		# Create a queue for BFS
		queue = []

		# Mark the source node as visited and enqueue it
		queue.append(s)
		visited[s] = True
		print queue
		while queue:

			# Dequeue a vertex from queue and print it
			s = queue.pop(0)
			print s,

			# Get all adjacent vertices of the dequeued
			# vertex s. If a adjacent has not been visited,
			# then mark it visited and enqueue it
			for i in self.graph[s]:
				if visited[i] == False:
					queue.append(i)
					visited[i] = True
	def get_self(self):
		print self.graph


# Driver code
# Create a graph given in the above diagram
g = Graph()
g.connections2graph(test, test2)
g.get_self()


sys.exit()
print "Following is Breadth First Traversal (starting from vertex 2)"
g.BFS(0)
print "\n"
g.BFS(1)
print "\n"
g.BFS(2)
print "\n"
g.BFS(3)


# This code is contributed by Neelam Yadav
