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
from skimage.morphology import skeletonize_3d

mito_prefix = "M_"
skeleton_suffix = "_skel"
binary_suffix = "_bin"
figure_suffix = "_fig.png"

def analyze(UID, read_path, write_path):
	mito = io.imread(read_path)
	z, x, y = mito.shape
	# stack_viewer(mito)
	# z = 2
	binary = np.zeros_like(mito)
	max_P_d = max_projection(mito)
	for layer in xrange(z):
		sel_elem = disk(1)
		layer_data = mito[layer,:,:]
		output1 = gamma_stabilize(layer_data,
									alpha_clean = 0.05,
									floor_method = 'min')
		output2 = smooth(output1,
							smoothing_px = 0.5,
							threshold = 1)
		m4 = median(output1, sel_elem)
		d = disk_hole(m4,
						radius = 2,
						pinhole = True)

		# USE ONLY FOR MITOS
		a8 = fft_ifft(m4, d)

		a9 = img_type_2uint8(a8, func = 'floor')

		asdf = median(a9, sel_elem)
		# Simple threshold
		scale_ratio = 3
		threshold = np.mean(asdf.flatten()) + np.std(asdf.flatten()) * scale_ratio
		# print "> Simple Thresholding threshold: {}".format(threshold)
		asdf[asdf <= threshold] = 0
		asdf[asdf > 0] = 1

		wow = label_and_correct(asdf, a9,
									min_px_radius = 2,
									max_px_radius = 100,
									min_intensity = 0,
									mean_diff = 10)
		wow[wow > 0] = 1

		binary[layer, :, :] = wow

	spooky = skeletonize_3d(binary)
	binary_projection = max_projection(binary)
	# montage_n_x((binary_projection, max_P_d))
	save_figure(max_P_d, mito_prefix + UID + "_maxP" + figure_suffix, write_path)
	save_figure(binary_projection, mito_prefix + UID + "_maxPB" + figure_suffix, write_path)
	save_data(spooky, mito_prefix + UID + skeleton_suffix, write_path)
	save_data(binary, mito_prefix + UID + binary_suffix, write_path)
	# return binary, spooky

if __name__ == "__main__":
	arguments = sys.argv
	analyze(arguments[-1])
