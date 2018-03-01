import sys
from skimage import io
import argparse
from lib.render import *
from lib.processing import *
from skimage.morphology import disk
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

	image = io.imread(read_path)
	z_dim, x_dim, y_dim = image.shape
	for slice_index in xrange(z_dim):
		slice_data = image[slice_index, :, :]
		noise_disk = disk(1)
		output1 = gamma_stabilize(slice_data, alpha_clean = 0.05)
		output2 = smooth_tripleD(output1, smoothing_px = 0.5, stdevs = 1.5)
		output3 = dilation(output2, disk(3))
		output3 = erosion(output3, disk(3))
		output4 = median(output3, noise_disk)
		output5 = img_type_2uint8(output4, func = 'floor')
		output6 = binarize_image(output5, _dilation = 0, feature_size = 25)
		output7 = improved_watershed(output6, output5, expected_separation = 20)
		# asdf = fft_ifft(output2, FFT_filter)
		montage_n_x((slice_data, output1, output2),( output3, output4, output5, output6, output7))
	# stack_viewer(image)





if __name__ == "__main__":
	main(sys.argv)
