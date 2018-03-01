from lib.render import *
import scipy.io
import sys
import argparse

'''
Given a path directory to a single image, script displays image on screen (2d and 3d compatible)
'''
def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for image visualization')
	parser.add_argument('-i', dest = 'image', help = 'Image path', required = True)
	options = vars(parser.parse_args())
	return options

def main(args):
	options = get_args(args)
	path = options['image']
	image = scipy.io.loadmat(path)['data']
	if len(image.shape) == 2:
		view_2d_img(image)
	elif len(image.shape) == 3:
		stack_viewer(image)
	else:
		print "Too many dimensions to resolve"


if __name__ == "__main__":
	main(sys.argv)
