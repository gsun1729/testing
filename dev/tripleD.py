import sys
from skimage import io
import argparse

def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for analyzing 3d Images without 2d compression')
	parser.add_argument('-r',
						dest = 'read_path',
						help = 'Raw data read directory',
						required = True)
	parser.add_argument('-w',
						dest = 'write_path',
						help = 'Results save directory',
						required = True)
	options = vars(parser.parse_args())
	return options


def main(args):
	options = get_args(args)
	read_dir = options['read_path']
	save_dir = options['write_path']




if __name__ == "__main__":
	main(sys.argv)
