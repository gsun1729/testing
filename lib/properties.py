import sys
import numpy as np

def global_max(img_2d):
	return np.amax(img_2d.flatten())

def global_min(img_2d):
	return np.amin(img_2d.flatten())

def properties(image):
	print ">Image Properties"
	print "Dimensions: {}".format(image.shape)
	print "Format: {}".format(image.dtype.name)
	print "Global Max: {}\nGlobal Min: {}".format(global_max(image), global_min(image))

if __name__ == "__main__":
	print "This file is not intended to be run on its own"
	sys.exit()
