import os
import re
def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.tif','', re.IGNORECASE)
					img_filelist.append((img_filename, img_file, os.path.join(current_location, img_file)))
	return img_filelist
