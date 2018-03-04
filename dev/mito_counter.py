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



def layer_comparator(image3D):
	equivalency_table = []
	zdim, xdim, ydim = image3D.shape
	kernel_dim = 2
	last_used_label = 0
	kernel = np.zeros((2,2,2))
	for z in range(1, zdim):
		for x in xrange(1, xdim):
			for y in xrange(1, ydim):
				print z,x,y
				kernel = image3D[z - kernel_dim + 1:z + 1,
									x - kernel_dim + 1:x + 1,
									y - kernel_dim + 1:y + 1]
				POI = kernel[1, 1, 1]
				if POI == 0:
					pass
				else:
					neighbors_1U = [kernel[0, 1, 1],
									kernel[1, 0, 1],
									kernel[1, 1, 0]]
					neighbors_2U = [kernel[0, 0, 1],
									kernel[0, 1, 0],
									kernel[1, 0, 0]]
					neighbors_3U = kernel[0, 0, 0]
					if any(neighbor != 0 for neighbor in neighbors_1U):
						print "asdf"


				#
				# print POI
			# L0 = layer0[x:x + kernel_dim, y:y + kernel_dim]
			# L1 = layer1[x:x + kernel_dim, y:y + kernel_dim]
			# KERNEL = np.concatenate((L0, L1))
			# return KERNEL
			# immediate_Right = L0[x, y + 1]
			# immediate_Bott = L0[x + 1, y]
			# immediate_Diag = L0[x + 1, y + 1]
			# top = L1[x, y]
			# top_right = L1[x, y + 1]
			# top_bot = L1[x + 1, y]
			# top_diag = L1[x + 1, y + 1]
			# if POI = 0:
			# 	pass
			# else:
			# 	if POI

	#
	# 		print POI
	# 		print L0
	# 		return
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
	b = np.zeros((10,10))
	a[3:8,4:7] = 1
	a[0,0] = 1
	a[0,1] = 1

	a2[5:9,6:9] = 1
	b[5:9,6:9] = 1

	stack = np.array([a,a2,b])
	print stack.shape
	print stack
	# print b
	neighbors = [0 ,2 ,1]
	print any(v == 0 for v in neighbors)
	q = [i for i in neighbors if i != 0]
	print q
	# layer_comparator(stack)
	# print [a[item]==0 for item in a]
	# q = all(v == 0 for v in row for row in a)

	# print any(0 in sublist for sublist in q)
