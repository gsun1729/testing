import sys
import scipy.io
import argparse
from lib.render import *
from lib.processing import *
from skimage.morphology import (disk, dilation, erosion)
from skimage.filters import rank
import scipy.signal
import cv2
# from lread_write import *

def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for analyzing 3d Images without 2d compression')
	parser.add_argument('-r',
						dest = 'read_path',
						help = 'Raw data read directory',
						required = True)
	parser.add_argument('-w',
						dest = 'write_path',
						help = 'Results save directory',
						required = False)
	options = vars(parser.parse_args())
	return options


def main(args):
	options = get_args(args)
	read_path = options['read_path']
	save_path = options['write_path']
	image = scipy.io.loadmat(read_path)['data']
	stack_viewer(image)

if __name__ == "__main__":
	main(sys.argv)
