import sys
import argparse
from render import *
from processing import *
from math_funcs import *
from properties import properties
from read_write import *
from skimage import io
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from skimage.morphology import (disk, dilation, watershed,
								closing, opening, erosion, skeletonize, medial_axis)
import matplotlib.pyplot as plt
cell_prefix = "C_"
data_suffix = "_dat"
projection_suffix = "_avgP"

def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for 3d segmenting mitochondria')
	parser.add_argument('-id',
						dest = 'UUID',
						help = 'Unique identifier (user generated)',
						required = True)
	parser.add_argument('-r',
						dest = 'read_path',
						help = 'read directory for data'
						required = True)
	parser.add_argument('-w',
						dest = 'write_path',
						help = 'write directory for results'
						required = True)

	options = vars(parser.parse_args())
	return options


def analyze(UID, read_path, write_path):
	try:
		options = get_args(sys.argv)
		read_path = options['read_path']
		write_path = options['write_path']
		UID = options['UUID']
	except:
		sys.exit()

	cell = io.imread(read_path)

	a1 = avg_projection(cell)
	a2 = gamma_stabilize(a1,
							alpha_clean = 1.3)
	# Remove noise
	a3 = smooth(a2,
					smoothing_px = 0.5,
					threshold = 1)
	noise_disk = disk(1)
	d1 = img_type_2uint8(a3, func = 'floor')
	d2 = median(d1, noise_disk)
	d3 = erosion(d2, noise_disk)
	d4 = median(d3, noise_disk)
	# Simple Thresholding
	mean, stdev = px_stats(d4)
	d4[d4 < mean + stdev] = 0
	d5 = img_type_2uint8(d4, func = 'floor')

	d6 = binarize_image(d5,
							_dilation = 0,
							feature_size = 25)
	d7 = binary_opening(d6, structure = disk(3).astype(np.int))
	d8 = binary_opening(d7).astype(np.int)
	# Recovery from earlier erosion
	d9 = dilation(d8, noise_disk)
	d10 = binary_opening(d9).astype(np.int)
	d11 = label_and_correct(d10, d5,
								min_px_radius = 7,
								max_px_radius = 200,
								min_intensity = 0,
								mean_diff = 10)
	d12 = improved_watershed(d11, d5,
								expected_separation = 2)
	d13 = rm_eccentric(d12,
						min_eccentricity = 0.68,
						max_area = 2500)
	# montage_n_x((d12,d13))
	write_stats(d12, d13, UID,
					"single_cell_stats.txt",
					read_path,
					write_path)

	save_data(a1, cell_prefix + UID + projection_suffix, write_path)
	save_data(d13, cell_prefix + UID + data_suffix, write_path)
	# return a16


if __name__ == "__main__":
	analyze(sys.argv)
