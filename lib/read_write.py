import os
import re
import scipy.io
import scipy.misc

img_suffix = ".tif"


def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.tif','', re.IGNORECASE)
					img_filelist.append((img_filename, img_file, current_location, os.path.join(current_location, img_file)))
	return img_filelist


def save_img(img, filename, img_type, file_directory):

	filename += img_type + img_suffix
	save_dir = os.path.join(file_directory, filename)
	scipy.misc.imsave(save_dir, img)
	print ">Image '{}' saved to '{}'".format(filename, save_dir)


def save_data(data, filename, data_type, file_directory):
	filename += data_type
	save_dir = os.path.join(file_directory, filename)
	scipy.io.savemat(save_dir, mdict={'data': data})
	# scipy.io.savemat(file_directory,)
	print ">Data '{}' saved to '{}'".format(filename, save_dir)
