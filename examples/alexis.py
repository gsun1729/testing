from skimage import io
import sys
from lib.render import *
from lib.processing import *
import lib.read_write as rw
from PIL import Image
from skimage import feature
from skimage.filters import threshold_local
from skimage.morphology import binary_dilation, binary_erosion
import numpy as np

suffix = ".tif"


def PROCESS_IMAGE(image_path):
	'''Reads in 4 channel image and implements operations across all channels
	Will only work with 2D images
	'''
	image = io.imread(image_path)
	GFP_raw = image[:, :, 0]
	RFP_raw = image[:, :, 1]
	DAPI_raw = image[:, :, 3]

	processed_dapi = PROCESS_BIG_PUNCTA(DAPI_raw)
	# processed_GFP = PROCESS_PUNCTA(GFP_raw)
	processed_RFP = PROCESS_PUNCTA(RFP_raw)



def PROCESS_PUNCTA(image):
	x, y = image.shape
	cleaned = gamma_stabilize(image,
								alpha_clean = 1,
								floor_method = 'min')
	sm = smooth(cleaned,
				smoothing_px = 1,
				threshold = 1)
 	converted = img_type_2uint8(sm, func = 'floor')
	flattened = converted.flatten()
	n_bins = int(2 * iqr(flattened) * (len(flattened) ** (1/3))) * 2
	print(n_bins)
	n, bin_edge = np.histogram(flattened, n_bins)
	peak = np.argmax(n)
	bin_midpt = (bin_edge[peak] + bin_edge[peak + 1]) / 2
	test_mask = converted > bin_midpt
	test_masked = converted * test_mask


	local_thresh = threshold_local(test_masked,
									block_size = 51,
									offset = -15)
	binary_local = test_masked > local_thresh

	# label individual elements and remove really small noise and background
	labeled_img = just_label(binary_local)

	montage_n_x((image, labeled_img))
	return labeled_img


def PROCESS_BIG_PUNCTA(image):
	x, y = image.shape
	cleaned = gamma_stabilize(image,
								alpha_clean = 7.5,
								floor_method = 'min')
	converted = img_type_2uint8(cleaned, func = 'floor')
	sel_elem = disk(2)
	median_filtered = median(converted, sel_elem)

	binary = median_filtered > 0

	binary_e = binary_erosion(binary, selem = disk(2), out = None)
	binary_d = binary_dilation(binary_e, selem = disk(2), out = None)


	corrected_slice = label_and_correct(binary_d, converted,
										min_px_radius = 5,
										max_px_radius = 100,
										min_intensity = 0,
										mean_diff = 11.9)
	montage_n_x((image, corrected_slice))
	return corrected_slice


print("test")
PROCESS_IMAGE("/home/gsun/Desktop/AT/MIP apj1/MAX_Apj1_30min42_003.tif")
