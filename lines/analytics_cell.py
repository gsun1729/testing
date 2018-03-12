import re
import sys
import os
# sys.path.insert(0, 'C:\\Users\\Gordon Sun\\Documents\\Github\\testing\\lib')
import lib.read_write as rw


before_HS = ['beforehs', 'before hs']
after_HS = ['hs', 'after hs', 'afterhs']
recovery = ['rec100', 're100', 'after recovery', 'recovery']


def extract_details(filename_path):
	HS_status_found = True

	B_grp, H_grp, R_grp = False, False, False
	well_found = True
	plate_found, ts_plate = True, True
	WT_grp, ry411_grp = False, False

	fpath = os.path.normpath(filename_path).lower()
	path_list = fpath.split("\\")
	filename = path_list[-1]
	set_path_list = set(path_list)

	# get Heat shock status
	if set_path_list.intersection(set(before_HS)):
		B_grp = True
	else:
		if set_path_list.intersection(set(after_HS)):
			H_grp = True
		elif set_path_list.intersection(set(recovery)):
			R_grp = True
		else:
			H_grp = True
			HS_status_found = False

	# Get Well ID
	try:
		well_ID = re.search('[a-h]{1}(\d+)(?!.*[a-h]{1}(\d+))', fpath, re.IGNORECASE).group()
	except AttributeError:
		try:
			well_ID = re.search('[a-h]{1}(_)(\d+)(?!.*[a-h]{1}(_)(\d+))', fpath, re.IGNORECASE).group()
		except AttributeError:
			well_ID = ''
			well_found = False

	# Get if WT_grp
	if "wt" in path_list[-1]:
		WT_grp = True
	elif "ry411" in path_list[-1]:
		ry411_grp = True

	# Get ts
	try:
		ts_ID = re.search('(?<!\w)(ts)(?!=\w)', fpath, re.IGNORECASE).group()
	except AttributeError:
		ts_ID = ''
		ts_plate = False

	# Get plate
	try:
		plate_ID = re.search('p(\d+)(?!.*p(\d+))', fpath, re.IGNORECASE).group()
		plate_ID = re.search('(\d+)', plate_ID, re.IGNORECASE).group()
	except AttributeError:
		try:
			plate_ID = re.search('(?=plate).*?(\d+)', fpath, re.IGNORECASE).group()
			plate_ID = re.search('(\d+)', plate_ID, re.IGNORECASE).group()
		except AttributeError:
			plate_ID = ''
			plate_found = False
	# print "B_grp: {}\nHS: {}\nREC: {}\nHS_FOUND: {}\nWELL_FOUND: {}\nWell_ID: {}\nWT_grp: {}\nRy: {}\nts_ID: {}\nts: {}\nplate_ID: {}\nplate_fount: {}\n".format(B_grp, H_grp, R_grp, HS_status_found, well_found, well_ID, WT_grp, ry411_grp, ts_ID, ts_plate, plate_ID, plate_found)
	if B_grp + H_grp + R_grp != 1:
		print "irregular sum"
		sys.exit()

	if B_grp:
		group = 'BeforeHS'
	elif H_grp:
		group = 'AfterHS'
	elif R_grp:
		group = 'Recovery'

	if ts_plate:
		plate_type = 'TS'
	elif WT_grp:
		plate_type = 'WT'
	elif ry411_grp:
		plate_type = 'RY411'
	else:
		plate_type = 'KO'

	return group, plate_type, plate_ID, well_ID


def column(matrix, i):
    return [row[i] for row in matrix]


def main(save_dir):
	# Creates the group data
	cell_pairs = rw.read_txt_file(os.path.join(save_dir, "analysis", "Cell_mito_UUID_Pairs.txt"))

	cell_stats = rw.read_txt_file(os.path.join(save_dir, "cell", "single_cell_stats.txt"))

	cell_grouped = open(os.path.join(save_dir, "cell_mito_groups.txt"), "w")
	cell_stats_grouped = open(os.path.join(save_dir, "single_cell_stats_grouped.txt"), "w")

	name_list = column(cell_pairs, 0)

	# Write group numbers to a file
	for cellUUID, mitoUUID, cell_fname, mito_fname, shared_path, _ in cell_pairs:
		group, plate_type, plate_ID, well_ID = extract_details(os.path.join(shared_path, cell_fname))

		cell_grouped.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(cellUUID,
																			mitoUUID,
																			cell_fname,
																			mito_fname,
																			group, 
																			plate_type,
																			plate_ID,
																			well_ID,
																			shared_path))

	cell_grouped.close()


	# add group numbers to the single stats dataset
	cell_grp_database = rw.read_txt_file(os.path.join(save_dir, "cell_mito_groups.txt"))
	n = 0
	for cell, UUID, cell_num, cell_Fnum, deleted, radius, area, perimeter, E, read_path in cell_stats:
		# print cell, UUID, cell_num, cell_Fnum, deleted, radius, area, perimeter, E
		try:
			location = name_list.index(UUID)
			HS_group, Plate_type, plate_num, well_num = cell_grp_database[location][4:8]
			mito_UUID = cell_grp_database[location][1]
		except ValueError:
			mito_UUID = "Not Found"
			HS_group = "Not Found"
			Plate_type = "Not Found"
			plate_num = "Not Found"
			well_num = "Not Found"

		cell_stats_grouped.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(cell,
																										UUID,
																										mito_UUID,
																										cell_num,
																										cell_Fnum,
																										deleted,
																										HS_group,
																										Plate_type,
																										plate_num,
																										well_num,
																										radius,
																										area,
																										perimeter,
																										E,
																										read_path))
	cell_stats_grouped.close()



if __name__ == "__main__":
	# test = "l:\\common\gordon jobs\\20180129 linhao tsdatabase\\p5g4\\before hs\\p111g41_1_w1488 laser.tif"
	main("L:\\Users\\gordon\\00000004 - Running Projects\\20180126 Mito quantification for Gordon\\20180306_results")
	# test2 = "L:\\Common\\Gordon Jobs\\20180129 Linhao Database\\2017-10-02 Magic plate 46\\After recovery\\_gTHIS IS ACTUAL G10_003_w1488 Laser.TIF"
	# test3 = "L:\\Common\\Gordon Jobs\\20180129 Linhao Database\\07112017 ts plate new part 2\\D12\\BeforeHS\\D12_2_w2561 Laser.TIF"
	# "L:\Common\Gordon Jobs\20180129 Linhao Database\p1 ts\A12\HS\\a12_4_w2561 Laser.TIF"
	# test5 = "L:\\Common\\Gordon Jobs\\20180129 Linhao Database\\2017-09-27 MAGIC plate 44 and 47\\_b8_003_w2561 Laser.TIF"

	# extract_details(test5)
