from skimage import io
import sys
from lib.render import *
from lib.processing import *
from PIL import Image
from skimage import feature
from skimage.filters import threshold_otsu, threshold_adaptive
import numpy as np
# def mask_px_stats(segmented_binary, labeled_trait):
# 	prin

def single(image_path):
	image = io.imread(image_path)


	sys.exit()
	DAPI = image[:, :, 0]
	YFP = image[:, :, 1]

	x, y, z = image.shape
	#
	DAPI = gamma_stabilize(DAPI, alpha_clean = 1.3)
	YFP = gamma_stabilize(YFP, alpha_clean = 1.3)
	#
	DAPI = smooth(DAPI)
	YFP = smooth(YFP)
	#
	sel_elem = disk(2)
	DAPI = median(DAPI, sel_elem)

	a5 = erosion(DAPI, selem = disk(1))
	a6 = median(a5, sel_elem)
	a7 = dilation(a6, selem = disk(1))
	a8 = img_type_2uint8(a7, func = 'floor')

	# mean, stdev = px_hist_stats_n0(a8)
	# a8[a8 <= mean - (stdev * 0.5)] = 0

	# view_2d_img(a8)
	# sys.exit()
	# d = disk_hole(a8, 500, pinhole = False)
	# a8_fft = fft_ifft(a8, d)

	a9 = binarize_image(a8, _dilation = 0, feature_size = 100)
	a10 = binary_fill_holes(a9).astype(int)
	# a11 = label_and_correct(a10, a8, min_px_radius = 10, max_px_radius = 500, min_intensity = 0, mean_diff = 10)
	a11 = label_and_correct(a10, a8, min_px_radius = 10, max_px_radius = 500, min_intensity = 0, mean_diff = 10)
	# a111 = label_and_correct(a10, a8, min_px_radius = 10, max_px_radius = 500, min_intensity = 0, mean_diff = 15)
	segmented_binary = improved_watershed(a11, YFP, expected_separation = 1)

	unlabeled_binary = np.zeros_like(segmented_binary)
	unlabeled_binary[segmented_binary >= 1] = 1

	YFP_masked = YFP * segmented_binary



def iterator():
	arguments = sys.argv
	read_dir = arguments[-1]

	image = single(read_dir)




if __name__	 == "__main__":
	iterator()
