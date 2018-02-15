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
	for layer in xrange(z):
		view_2d_img(mito[layer,:,:])
	stack_viewer(mito)


if __name__ == "__main__":
	arguments = sys.argv

	analyze(arguments[-1])
