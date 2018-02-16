import os
import re
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from scipy.misc import imsave

img_suffix = ".tif"


def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.TIF','')
					img_filelist.append((img_filename, current_location, os.path.join(current_location, img_file)))
	return img_filelist
	# for img_fn, img_f, current_loc, full_path in img_filelist:
	# 	return img_fn, img_f, current_loc, full_path


# def save_img_data(img, filename, img_type, file_directory):
#
# 	filename += img_type + img_suffix
# 	save_dir = os.path.join(file_directory, filename)
# 	scipy.misc.imsave(save_dir, img)
# 	print "> Image '{}' saved to '{}'".format(filename, save_dir)

# def save_fig(img)

def save_data(data, filename, data_type, file_directory):
	filename += data_type
	save_dir = os.path.join(file_directory, filename)
	scipy.io.savemat(save_dir, mdict={'data': data})
	# scipy.io.savemat(file_directory,)
	print "> Image Data '{}' saved to '{}'".format(filename, file_directory)

def save_figure(fig, name, path):
	imsave(os.path.join(path,name), fig)
	print "> Image Figure '{}' saved to '{}'".format(name, path)
