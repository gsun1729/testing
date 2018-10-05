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
								closing, opening, erosion, medial_axis)
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
cell_prefix = "C_"
data_suffix = "_dat"
projection_suffix = "_avgP"

def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for 3d segmenting cells')
	parser.add_argument('-id',
						dest = 'UUID',
						help = 'Unique identifier (user generated)',
						required = True)
	parser.add_argument('-r',
						dest = 'read_path',
						help = 'read directory for data',
						required = True)
	parser.add_argument('-w',
						dest = 'write_path',
						help = 'write directory for results',
						required = True)

	options = vars(parser.parse_args())
	return options


def analyze(UID, read_path, write_path):
	cell = io.imread(read_path)
	noise_disk = disk(1)
	noise_processing = avg_projection(cell)
	noise_processing = gamma_stabilize(noise_processing, alpha_clean = 1.3)
	for i in xrange(1):
		# Remove noise
		noise_processing = smooth(noise_processing,
									smoothing_px = 0.5,
									threshold = 1)

	noise_processing = img_type_2uint8(noise_processing, func = 'floor')
	noise_processing = erosion(noise_processing, noise_disk)
	noise_processing = median(noise_processing, noise_disk)

	# d4 = median(d3, noise_disk)
	# Simple Thresholding
	d14 = smooth(noise_processing, smoothing_px = 4, threshold = 1)
	test = threshold_local(noise_processing,
									block_size = 31,
									offset = 0)
	# mean, stdev = px_stats(noise_processing)
	# noise_processing[noise_processing < mean + stdev] = 0
	# d5 = img_type_2uint8(noise_processing, func = 'floor')

	temp_binary = d14 > test *0.9
	view_2d_img(temp_binary)
	wow = improved_watershed(temp_binary, d14,
							expected_separation = 3)
	wow2 = rm_eccentric(wow,
						min_eccentricity = 0.7,
						max_area = 2500)
	# wow = label_and_correct(temp_binary, noise_processing, min_px_radius = 7,
	# 						max_px_radius = 200,
	# 						min_intensity = 0,
	# 						mean_diff = 10)
	montage_n_x((d14,noise_processing,  temp_binary, wow, wow2))
	raise Exception
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
	montage_n_x((a1,d12,d13))
	raise Exception
	write_stats(d12, d13, UID,
					"single_cell_stats.txt",
					read_path,
					write_path)

	save_data(a1, cell_prefix + UID + projection_suffix, write_path)
	save_data(d13, cell_prefix + UID + data_suffix, write_path)
	# return a16


if __name__ == "__main__":
		try:
			options = get_args(sys.argv)
			read_path = options['read_path']
			write_path = options['write_path']
			UID = options['UUID']
			analyze(sys.argv)
		except:
			sys.exit()
