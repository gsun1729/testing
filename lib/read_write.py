import os
import re
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from scipy.misc import imsave
import skimage.io
import uuid
img_suffix = ".tif"


def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.TIF','')
					unique_ID = str(uuid.uuid4().hex)
					path_difference = os.path.relpath(current_location, root_directory)
					img_filelist.append((unique_ID,
										img_filename,
										img_file,
										path_difference,
										current_location,
										os.path.join(current_location, img_file)))
	return img_filelist


def save_data(data, filename, write_directory):
	save_dir = os.path.join(write_directory, filename)
	scipy.io.savemat(save_dir, mdict={'data': data})
	# scipy.io.savemat(write_directory,)
	print "> Image Data '{}' saved to '{}'".format(filename, write_directory)


def save_figure(fig, name, write_directory):
	imsave(os.path.join(write_directory,name), fig)
	print "> Image Figure '{}' saved to '{}'".format(name, write_directory)


def filepath2name(filepath):
	if filepath[0] == ".":
		filepath = list(filepath	)
		filepath[0] = ""
		filepath = "".join(filepath)
	filepath = filepath.replace("\\","_")
	filepath = filepath.replace(" ","-")
	return filepath


# def channel_separator(multichannel_img_path):
# 	image = skimage.io.imread(multichannel_img_path)
# 	print image.shape
