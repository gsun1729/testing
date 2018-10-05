import sys
from render import *
from processing import *
from math_funcs import *
from properties import properties
from read_write import *
from skimage import io
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import (disk, dilation, watershed,
								closing, opening, erosion, skeletonize, medial_axis)
from skimage.morphology import skeletonize_3d
from skimage.filters import threshold_local

mito_prefix = "M_"
skeleton_suffix = "_skel"
binary_suffix = "_bin"
figure_suffix = "_fig.png"

def analyze(UID, read_path, write_path):
	# Script is designed to handle 512 by 512 sized images
	mito = io.imread(read_path)
	z, x, y = mito.shape
	binary = np.zeros_like(mito)
	max_P_d = max_projection(mito)
	for layer in xrange(z):
		sel_elem = disk(1)
		layer_data = mito[layer,:,:]
		output1 = gamma_stabilize(layer_data,
									alpha_clean = 1,
									floor_method = 'min')
		output2 = smooth(output1,
							smoothing_px = 2,
							threshold = 1)
		median_filtered = median(output1, sel_elem)
		fft_filter_disk = disk_hole(median_filtered,
									radius = 5,
									pinhole = True)
		# Remove High frequency noise from image
		FFT_Filtered = fft_ifft(median_filtered, fft_filter_disk)
		# Convert image to 8 bit for faster processing
		image_8bit = img_type_2uint8(FFT_Filtered, func = 'floor')
		# Run local thresholding and binarization
		local_thresh = threshold_local(image_8bit,
										block_size = 31,
										offset = -15)
		binary_local = image_8bit > local_thresh
		# label individual elements and remove really small noise and background
		corrected_slice = label_and_correct(binary_local, image_8bit,
												min_px_radius = 1,
												max_px_radius = 100,
												min_intensity = 0,
												mean_diff = 15)
		corrected_slice[corrected_slice > 0] = 1
		# montage_n_x((image_8bit, binary_adaptive,  binary_adaptive3, corrected_slice))
		binary[layer, :, :] = corrected_slice
	print 'OK\n'
	spooky = skeletonize_3d(binary)
	binary_projection = max_projection(binary)
	save_figure(max_P_d, mito_prefix + UID + "_maxP" + figure_suffix, write_path)
	save_figure(binary_projection, mito_prefix + UID + "_maxPB" + figure_suffix, write_path)
	save_data(spooky, mito_prefix + UID + skeleton_suffix, write_path)
	save_data(binary, mito_prefix + UID + binary_suffix, write_path)

if __name__ == "__main__":
	# Run as standalone on single image
	arguments = sys.argv
	analyze(arguments[-1])
