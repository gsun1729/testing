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
	mito = io.imread(input_image_pathway)
	z, x, y = mito.shape
	# stack_viewer(mito)
	# z = 2
	binary = np.zeros_like(mito)
	for layer in xrange(z):
		sel_elem = disk(1)
		layer_data = mito[layer,:,:]
		output1 = gamma_stabilize(layer_data, alpha_clean = 0.05)
		output2 = smooth(output1)
		m4 = median(output2, sel_elem)
		# m5 = erosion(m4, selem = disk(1))
		# m6 = median(m5, sel_elem)
		# a7 = dilation(m6, selem = disk(1))
		d = disk_hole(m4, 1, pinhole = True)

		# USE ONLY FOR MITOS
		a8 = fft_ifft(m4, d)

		a9 = img_type_2uint8(a8, func = 'floor')

		asdf = median(a9, sel_elem)
		# Simple threshold
		scale_ratio = 10
		threshold = np.mean(asdf.flatten()) * scale_ratio
		print "> Simple Thresholding threshold: {}".format(threshold)
		asdf[asdf <= threshold] = 0
		asdf[asdf > 0] = 1

		wow = label_and_correct(asdf, a9, min_px_radius = 3)
		wow[wow > 0] = 1
		# montage_n_x((a9, wow))
		binary[layer,:,:] = wow
		# montage_n_x((q, a10))
		# montage_n_x((layer_data, output1, output2),(m4, a8, a9, asdf))
	stack_viewer(binary)
	return binary
# def apply_over_layers(function())


if __name__ == "__main__":
	arguments = sys.argv
	analyze(arguments[-1])
	# q = gen_a(1,5)
	# print mix(multiply(*a),q)
