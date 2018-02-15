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

def analyze(input_image_pathway):
	cell = io.imread(input_image_pathway)
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
	a9 = binarize_image(a8, _dilation = 0, heterogeity_size = 10, feature_size = 2)
	a10 = binary_fill_holes(a9).astype(int)

	a11 = label_and_correct(a10, a8, min_px_radius = 10)
	a12 = remove_element_bounds(a11, lower_area = 550, upper_area = 3000)
	a13 = measure.find_contours(a12, level = 0.8, fully_connected = 'low', positive_orientation = 'low')
	# Split and label double cells
	a14 = cell_split(a12, a13, min_area = 100, max_area = 3500, min_peri = 100, max_peri = 1500)
	# REMOVE ANYTHING HUGE
	a100 = remove_element_bounds(a14, lower_area = 10, upper_area = 1800)


	# Update contours
	a15 = measure.find_contours(a100, level = 0.8, fully_connected = 'low', positive_orientation = 'low')
	# Plot block
	render_contours(a100, a15)
	montage_n_x((a1, a2, a3, a4, a5, a6), (a7, a9, a10, a11, a12, a14, a100))
	view_2d_img(a100)

	return a14

if __name__ == "__main__":
	arguments = sys.argv

	analyze(arguments[-1])
