from lib.render import *
import scipy.io
import sys
import argparse
from scipy import ndimage as ndi
'''
Given a path directory to a single image, script displays image on screen (2d and 3d compatible)
'''
def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for image visualization')
	parser.add_argument('-i', dest = 'image', help = 'Image path', required = True)
	options = vars(parser.parse_args())
	return options

def main(args):
	options = get_args(args)
	path = options['image']
	image = scipy.io.loadmat(path)['data']

	z, x, y = image.shape
	image[image > 0] = 1
	stack_viewer(image)
	# for stack_index in xrange(z - 1):
	# 	labeled_stack, object_no = ndi.label(image[stack_index, :,:], structure = np.ones((3, 3)))
	# 	view_2d_img(labeled_field)



def layer_comparator(layer0, layer1):
	equivalency_table = []
	xdim, ydim = layer0.shape
	kernel_dim = 2
	for x in xrange(xdim - kernel_dim + 1):
		for y in xrange(ydim - kernel_dim + 1):
			L0 = layer0[x:x + kernel_dim, y:y + kernel_dim]
			L1 = layer1[x:x + kernel_dim, y:y + kernel_dim]
			print L0
			# sys.exit()
			# L0 = layer0[x:]
			# L0_p1 = layer0(x, y)
			# L0_p2 = layer0(x, y + 1)
			# L0_p3 = layer0(x + 1, y)
			# L0_p4 = layer0(x + 1, y + 1)
			# L1_p1 = layer1(x, y)
			# L1_p2 = layer1(x, y + 1)
			# L1_p3 = layer1(x + 1, y)
			# L1_p4 =


if __name__ == "__main__":
	a = np.zeros((10,10))
	a2 = np.zeros((10,10))
	a[3:8,4:7] = 1
	a[0,0] = 1
	a2[5:9,6:9] = 1
	b = np.zeros((3,10))
	print a
	print a2
	print b
	a = [0,1,2,3,0,4,5]
	print [a[item]==0 for item in a]
	# q = all(v == 0 for v in row for row in a)
	# layer_comparator(a,a2)
	# main(sys.argv)
