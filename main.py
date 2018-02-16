
import sys
sys.path.insert(0, '.\\lib')
sys.path.insert(0, '.\\lines')
import os
import cell_line
import mito_line

from render import *
from read_write import *
from skimage import io
# from render import *
from processing import *
# from math_funcs import *
# from properties import properties
# from read_write import *
#
# from skimage import measure
# from scipy.ndimage.morphology import binary_fill_holes
# from skimage.morphology import (disk, dilation, watershed,
# 								closing, opening, erosion, skeletonize, medial_axis)
# # from skimage.segmentation import random_walker
# # from skimage.restoration import denoise_bilateral, estimate_sigma
# import scipy.signal as ss
# from sklearn.preprocessing import normalize
#
# from scipy import ndimage as ndi, stats
# from scipy.ndimage import gaussian_filter
# from skimage.feature import peak_local_max
# from skimage.filters import median, rank, threshold_otsu, laplace
# from math_funcs import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def main():
	os.system('cls' if os.name == 'nt' else 'clear')
	options = sys.argv
	if len(options) != 4:
		print "> Wrong # of arguments, please provide input in the form \n> python main.py read_directory write_directory\n"
		for num, item in enumerate(options):
			print "> \tArg{}: {}\n".format(num, item)
		sys.exit()
	else:
		root_read_dir = options[-3]
		save_dir_CELL = options[-2]
		save_dir_MITO = options[-1]
		print "> Parent Read Directory : {}\n".format(root_read_dir)
		print "> CELL Save Directory : {}\n".format(save_dir_CELL)
		print "> MITO Save Directory : {}\n".format(save_dir_MITO)

	filenames = get_img_filenames(root_read_dir)
	# print "\nFiles to be processed:\n"
	# for img_name, img_loc, img_path in filenames:
	# 	print img_path
	# sys.exit()
	# root = ".\\data\\generated"
	for img_name, img_loc, img_path in filenames:
		print "> ==========================================================================================\n"
		print "\n> Currently Processing : {}\n".format(img_name)
		if '1488' in img_name:
			print "> Image ID: 1488: Cell TD\n"
			# blockPrint()
			cell_line.analyze(img_name, img_loc, img_path, save_dir_CELL)
			# enablePrint()
		elif '2561' in img_name:
			print "> Image ID: 2561: Mitochondria\n"
			# blockPrint()
			mito_line.analyze(img_name, img_loc, img_path, save_dir_MITO)
			# enablePrint()
	# sys.exit()




#
# print 'This will print'
#
#
# print "This won't"
#
#
# print "This will too"
			# print os.path.join(img_name, img_filename)
	# cell_line.analyze(img_name, img_loc, img_path)
	# 	# elif '2561' in img_name:
	# 	# 	print "mito"
	# 	# else:
	# 	# 	print "whatdo"
	#
	#
	# sys.exit()
	# cell = ".\\data\\hs\\P11B3_2_w1488 Laser.TIF"
	# mito = ".\\data\\hs\\P11B3_2_w2561 Laser.TIF"
	# cell2 = ".\\data\\_hs\\P45F12_3_w1488 Laser.TIF"
	# mito2 = ".\\data\\_hs\\P45F12_3_w2561 Laser.TIF"
	# cell3 = ".\\data\\hs\\P26G1_1_w1488 Laser.TIF"
	# mito3 = ".\\data\\hs\\P26G1_1_w2561 Laser.TIF"
	# cell4 = ".\\data\\_hs\\P42A12_1_w1488 Laser.TIF"
	# mito4 = ".\\data\\_hs\\P42A12_1_w2561 Laser.TIF"
	# # # Bad Images
	# # cellb = ".\\data\\hs\\P34A12_3_w1488 Laser.TIF"
	# # mitob = ".\\data\\hs\\P34A12_2_w2561 Laser.TIF"
	#
	# cell_line.analyze(cell3)
	# # binary_map = mito_line.analyze(mito)
	# # skeleton3d = skeletonize_3d(binary_map)
	# # stack_viewer(skeleton3d)
	# sys.exit()

if __name__ == "__main__":
	main()
