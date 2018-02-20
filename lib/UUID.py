import os

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


def uuid_present(UUID, lookupTable):
	UUID_index = -1
	UUID_found = False
	for row_index, row in enumerate(lookupTable):
		if row[0] == UUID:
			UUID_found = True
			UUID_index = row_index
			break
	return UUID_index, UUID_found


def filename2info(filename, lookupTable):
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
	input_info = filename2info(filename, lookupTable)
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


def create_pairTable(filelist, lookupTable):
	UUID_pairs = []
	for row in filelist:
		input_info = filename2info(row, lookupTable)
		partner_info = get_partner(row, lookupTable)
		UUID_pairs.append([input_info[0], partner_info[0], input_info[2], partner_info[2], input_info[-2]])
	return UUID_pairs


def create_densepairTable(filelist, lookupTable):
	UUID_pairs = []
	for row in filelist:
		input_info = filename2info(row, lookupTable)
		partner_info = get_partner(row, lookupTable)
		UUID_pairs.append([input_info[0], partner_info[0]])
	return UUID_pairs


def write_list_txt(location, filename, array):
	writefile = open(os.path.join(location, filename), 'w')
	for row in array:
		for row_ele in xrange(len(row)):
			writefile.write(row[row_ele]+"\t")
		writefile.write("\n")
	writefile.close()
