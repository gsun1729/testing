
import sys
sys.path.insert(0, '.\\lib')
sys.path.insert(0, '.\\lines')
import os, errno
import cell_line
import mito_line
from render import *
from read_write import *
from skimage import io
from processing import *
import time, string, uuid
import argparse


def blockPrint():
	sys.stdout = open(os.devnull, 'w')


def enablePrint():
	sys.stdout = sys.__stdout__


def get_args(args):
	parser = argparse.ArgumentParser(description = 'Script for analyzing mitochondria skeletonization')
	parser.add_argument('-r', dest = 'read_dir', help = 'Raw data read directory', required = True)
	parser.add_argument('-w', dest = 'save_dir', help = 'Save directory for segmentation and skeletonization data', required = True)

	options = vars(parser.parse_args())
	return options

def main(args):
	os.system('cls' if os.name == 'nt' else 'clear')


	options = get_args(args)
	# if len(options) != 4:
	# 	print "> Wrong # of arguments, please provide input in the form \n> python main.py read_directory write_directory\n"
	# 	for num, item in enumerate(options):
	# 		print "> \tArg{}: {}\n".format(num, item)
	# 	sys.exit()
	# else:
	print options
	root_read_dir = options['read_dir']
	save_dir = options['save_dir']
	# save_dir_CELL = options[-2]
	# save_dir_MITO = options[-1]
	# 	print "> Parent Read Directory : {}\n".format(root_read_dir)
	# 	print "> CELL Save Directory : {}\n".format(save_dir_CELL)
	# 	print "> MITO Save Directory : {}\n".format(save_dir_MITO)
	print "> Parent Read Directory : {}\n".format(root_read_dir)
	print "> Save Directory : {}\n".format(save_dir)

	save_dir_cell = os.path.join(save_dir, 'cell')
	save_dir_mito = os.path.join(save_dir, 'mito')
	save_dir_anal = os.path.join(save_dir, 'analysis')

	mkdir_check(save_dir_cell)
	mkdir_check(save_dir_mito)
	mkdir_check(save_dir_anal)



	start = time.time()
	filenames = get_img_filenames(root_read_dir)
	num_images = len(filenames)
	end = time.time()
	print "> {} images detected, time taken: {}".format(num_images, end - start)


	mito_stats = []
	cell_stats = []

	print "> Processing IDs saved here: {}\n".format(save_dir)


	file_list_ID = open(os.path.join(save_dir, "filename_list.txt"),'w')
	for UID, img_name, img_fname, path_diff, img_loc, img_path in filenames:
		file_list_ID.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(UID, img_name, img_fname, path_diff, img_loc, img_path))
	file_list_ID.close()



	img_num = 1
	for UID, img_name, img_fname, path_diff, img_loc, img_path in filenames:
		print "> ==========================================================================================\n"
		print "\n> Currently Processing : {}\n".format(img_name)
		print "> \tImage Unique ID: {}\n".format(UID)
		print "> \tImage/Total Number of Images: {}/{}\n".format(img_num, num_images)
		if '1488' in img_name:
			print "> Image ID: 1488 - Cell TD\n"
			# blockPrint()
			start = time.time()
			cell_line.analyze(UID, img_path, save_dir_cell)
			end = time.time()
			# enablePrint()
			print "> Time to Compete: {}".format(end - start)
			mito_stats.append(end - start)
			img_num += 1

		elif '2561' in img_name:
			print "> Image ID: 2561 - Mitochondria\n"
			# blockPrint()
			start = time.time()
			mito_line.analyze(UID, img_path, save_dir_mito)
			end = time.time()
			# enablePrint()
			print "> Time to Compete: {}".format(end - start)
			cell_stats.append(end - start)
			img_num += 1



	print "> ==========================================================================================\n"
	print "> Prelim Analysis completed"
	save_data(mito_stats, "mito_processing_RT", save_dir)
	save_data(cell_stats, "cell_processing_RT",  save_dir)

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
	main(sys.argv)
