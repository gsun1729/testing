import sys

sys.path.insert(0, '.\\lib')
import os
import re
from PIL import Image
from skimage import io
from render import stack_viewer as sv


def split_and_trim(path_prefix, main_root):
	"""
	helper function for OS Path trimming routine that accounts for the trailing separator

	:param path_prefix: [str]
	:param main_root: [str]
	:return:[list]
	"""
	trim_length = len(main_root)
	print os.sep, main_root[-1]
	if main_root[-1] != os.sep:
		trim_length += 1

	return path_prefix[trim_length:].split(os.sep)



def get_img_filenames(root_directory):
	matched_images = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.TIF' in img_file or '.tif' in img_file) and '_thumb_' not in img_file:
					img_filename = img_file.replace('.tif','', re.IGNORECASE)
					matched_images.append((img_filename, img_file, os.path.join(current_location, img_file)))
	return matched_images

def main():
	root = ".\\Images\\generated"
	print get_img_filenames(root)
	a = io.imread(".\\Images\\linhao\\hs\\P34A12_1_w1488 Laser.TIF")
	print a.shape
	sv(a)

if __name__ == "__main__":
	main()
