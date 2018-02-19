import sys
sys.path.insert(0, 'C:\\Users\\Gordon Sun\\Documents\\Github\\testing\\lib')
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
import matplotlib.pyplot as plt

cell_prefix = "C_"
data_suffix = "_dat"
fig_suffix = "_fig.png"

def analyze(UID, read_path, write_path):
	cell = io.imread(read_path)
	sel_elem = disk(2)
	a1 = max_projection(cell)
	a2 = gamma_stabilize(a1, alpha_clean = 1.3)
	# Remove noise
	a3 = smooth(a2)
	a4 = median(a3, sel_elem)
	a5 = erosion(a4, selem = disk(1))
	a6 = median(a5, sel_elem)
	a7 = dilation(a6, selem = disk(1))
	a8 = img_type_2uint8(a7, func = 'floor')

	# Binarization
	a9 = binarize_image(a8, _dilation = 0, feature_size = 2)
	a10 = binary_fill_holes(a9).astype(int)
	a11 = label_and_correct(a10, a8, min_px_radius = 10)
	a12 = improved_watershed(a11, intensity = 10, expected_separation = 10)
	a13 = rm_eccentric(a12, min_eccentricity = 0.4, max_area = 1600)
	# montage_n_x((a11, a12,a13))

	# montage_n_x((a11,a12,a1))
	# a200 = measure.find_contours(a101, level = 0.8, fully_connected = 'low', positive_orientation = 'low')

	# a11 = label_and_correct(a10, a8, min_px_radius = 10)
	# a12 = remove_element_bounds(a11, lower_area = 550, upper_area = 3000)
	# a13 = measure.find_contours(a12, level = 0.8, fully_connected = 'low', positive_orientation = 'low')
	# Split and label double cells
	# a14 = cell_split(a12, a13, min_area = 100, max_area = 3500, min_peri = 100, max_peri = 1500)
	# REMOVE ANYTHING HUGE

	# a100 = remove_element_bounds(a14, lower_area = 10, upper_area = 1800)


	# eccentricity check
	# a15 = measure.find_contours(a100, level = 0.8, fully_connected = 'low', positive_orientation = 'low')
	# Final eccentricity cleanup
	# a16 = rm_eccentric(a100, a15, eccentricity = 0.4)

	# a17 = remove_element_bounds(a16, lower_area = 100, upper_area = 1800)


	# Plot block
	# render_contours(a100, a15)
	# montage_n_x((a1, a2, a3, a4, a5, a6,a7), (a9, a10, a101, a200))
	# view_2d_img(a100)
	# view_2d_img(a16, save = True)



	save_figure(a13, cell_prefix + UID + fig_suffix, write_path)
	save_figure(a1, cell_prefix + UID + "_maxP" + fig_suffix, write_path)
	save_data(a13, cell_prefix + UID + data_suffix, write_path)
	# return a16


if __name__ == "__main__":
	arguments = sys.argv

	analyze(arguments[-1])
