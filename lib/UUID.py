import os
global lookupTable

def get_UUID(filename):
	'''
	Given a string filename in the form X_UUID_XXX.XXX
	Return the UUID element in the filename
	:param filename: <str> of the filename in question
	:return: UUID <str> fraction of the input filename
	'''
	split_filename = filename.split('_')
	return split_filename[1]


def get_file_datatype(filename):
	'''
	Given a string filename in the form X_UUID_XXX.TYPE
	Return the TYPE element in the filename
	:param filename: <str> of the filename in question
	:return: type <str> fraction of the input filename
	'''
	split_filename = filename.split('_')
	return split_filename[-1]


def get_M_C(filename):
	'''
	Given a string filename in the form MC_UUID_XXX.TYPE
	Return the MC element in the filename
	:param filename: <str> of the filename in question
	:return: M or C <str> fraction of the input filename, single letter string
	'''
	split_filename = filename.split('_')
	return split_filename[0]


def read_UUID_file(location):
	'''
	Reads the lookuptable generated from the main part of the algorithm, strips any new lines and tabs
	:param location: directory LUT is located in, .txt file
	:return: <list> of <lists> which includes data from the LUT file.
	'''
	file_object = open(location, 'r')
	content = file_object.readlines()
	file_object.close()
	lookupTable = [x.strip('\n').split('\t') for x in content]
	return lookupTable

# Replaced by filename2info
# def uuid_present(UUID, lookupTable):
# 	'''
# 	Determines whether or not a UUID is present in the lookupTable
# 	:param UUID: UUID query
# 	:param lookupTable: Table to look for UUID in, list of lists
# 	:return: <int> index where the UUID was found (-1) if unfound, and <bool> if UUID was found
# 	'''
# 	UUID_index = -1
# 	UUID_found = False
# 	for row_index, row in enumerate(lookupTable):
# 		if row[0] == UUID:
# 			UUID_found = True
# 			UUID_index = row_index
# 			break
# 	return UUID_index, UUID_found


def filename2info(filename, lookupTable):
	'''
	Given a filename (MC_UUID_TYPE), retrieve the UUID and look for its presence in the LUT
	:param filename: complete filename in format MC_UUID_TYPE
	:param lookupTable: LUT to search in
	:return: if UUID is found, returns the row that the element was found in, does not return anything if nothing is found
	'''
	UUID = get_UUID(filename)
	found = False
	for row in lookupTable:
		if UUID == row[0]:
			found = True
			return row
	if found == False:
		assert "UUID ({}, {}) not found in LUT".format(UUID, filename)


def get_partner(filename, lookupTable):
	'''
	Given a UUID for a cell or mitochondria image, determine the UUID for the mitochondria or cell data partner
	:param filename: original filename of the file being queried (full filename)
	:param lookupTable: table to look up query for the filename
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
	'''

	'''
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
	'''
	Given an array, write data to filename at location.
	:param location: save directory for output file
	:param filename: name of the output file
	:param array: data to be saved to file.
	'''
	writefile = open(os.path.join(location, filename), 'w')
	for row in array:
		for row_ele in xrange(len(row)):
			writefile.write(row[row_ele]+"\t")
		writefile.write("\n")
	writefile.close()
