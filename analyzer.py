import sys
sys.path.insert(0, '.\\lib')
from render import *
from read_write import *
import scipy.io

def get_UUID(filename):
	split_filename = filename.split('_')
	return split_filename[1]


def get_file_datatype(filename):
	split_filename = filename.split('_')
	return split_filename[-1]


def get_M_C(filename):
	split_filename = filename.split('_')
	return split_filename[0]


def read_UUID_file(location):
	file_object = open(location, 'r')
	content = file_object.readlines()
	file_object.close()
	return [x.strip('\n').split('\t') for x in content]


def get_img_filenames(root_directory):
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if ('.mat' in img_file or '.mat' in img_file) and '_thumb_' not in img_file:
					img_filelist.append(img_file)
	return img_filelist


def verify_shape(img_2d, stack_3d):
	z3, x3, y3 = stack_3d.shape
	x2, y2 = img_2d.shape
	if x2 == x3 and y2 == y3:
		return True
	else:
		return False


def uuid_present(UUID, lookupTable):
	UUID_index = -1
	UUID_found = False
	for row_index, row in enumerate(lookupTable):
		if row[0] == UUID:
			UUID_found = True
			UUID_index = row_index
			break
	return UUID_index, UUID_found


def uuid2info(filename, lookupTable):
	UUID = get_UUID(filename)
	found = False
	for row in lookupTable:
		if UUID == row[0]:
			found = True
			return row
	if found == False:
		print "UUID ({}, {}) not found in LUT".format(UUID, filename)


def get_partner(filename, lookupTable):
	'''
	Given a UUID for a cell or mitochondria image, determine the UUID for the mitochondria or cell data partner
	'''
	attribute_type = get_M_C(filename)
	input_info = uuid2info(filename, lookupTable)
	if input_info == None:
		return
	else:
		input_fname = input_info[2]
		shared_loc = input_info[-2]
		if attribute_type == "M":
			partner_fname = input_fname.replace("2561", "1488")
		elif attribute_type == "C":
			partner_fname = input_fname.replace("1488", "2561")
		for row_num, row in enumerate(lookupTable):
			if partner_fname == row[2] and shared_loc == row[-2]:
				return row



def main():
	UID_file_loc = "L:\\Users\\gordon\\00000004 - Running Projects\\20180126 Mito quantification for Gordon\\results\\20180218_2 Run\\isolate\\filename_list.txt"
	root_read_dir = os.path.dirname(UID_file_loc)
	cell_data_dir = os.path.join(root_read_dir, "cell")
	mito_data_dir = os.path.join(root_read_dir, "mito")

	cell_filelist = get_img_filenames(cell_data_dir)
	mito_filelist = get_img_filenames(mito_data_dir)
	UUID_data = read_UUID_file(UID_file_loc)

	test = cell_filelist[0]
	test2 = test.replace("2","3")
	print test
	print test2

	a = get_partner(test2, UUID_data)
	print a
	a = get_partner(test, UUID_data)
	print a

	for row in UUID_data:
		print row

	for x, row in enumerate(UUID_data):
		if x%2 == 0:
			UUID_data[x] = []
	print "============"
	for row in UUID_data:
		print row

	# print cell_filelist
	# print test
	# print info
	# print get_partner(test, UUID_data)
	#
	#
	# print cell_filelist
	# print mito_filelist

	# cell_img = scipy.io.loadmat(os.path.join(cell_data_dir, cell_filelist[0]))
	# mito_stack = scipy.io.loadmat(os.path.join(mito_data_dir, mito_filelist[0]))
	# cell_img = cell_img['data']
	# mito_stack = mito_stack['data']
	# print cell_img.shape
	# print mito_stack.shape

	# test = "M_4e04f94102c544b0b6e2977eda957329_bin.mat"
	# print get_UUID(test)
if __name__ == "__main__":
	main()
