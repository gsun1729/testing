import re
import sys
import os
# sys.path.insert(0, 'C:\\Users\\Gordon Sun\\Documents\\Github\\testing\\lib')
import lib.read_write as rw


def column(matrix, i):
    return [row[i] for row in matrix]

'''
Script objective is to correlate mitochondria with cell statistics
ASSUMES SORTED DATA INPUT
'''
def main(save_dir):
	# Creates the group data
	# File with the UUID CEll and mitochondria groups and single cell data
	single_cell_stats = rw.read_txt_file(os.path.join(save_dir, "single_cell_stats_grouped.txt"))
	# File with mitochondria data
	mito_stats = rw.read_txt_file(os.path.join(save_dir, "mitochondria_statistics.txt"))
	# indexing of mito names for faster search
	name_list = column(single_cell_stats, 2)


	MASTER_SHEET = open(os.path.join(save_dir, "MASTER_RESULTS.txt"), "w")
	for cell_UUID, mito_UUID, cell_filename, mito_filename, filepath, cell_num, mito_num, volume, nTriangles, surface_area in mito_stats:
		location = name_list.index(mito_UUID)
		# print cell_UUID, mito_UUID, cell_num

		for start_search in xrange(location, len(single_cell_stats)):
			SC_cell_num = single_cell_stats[start_search][4]
			# print SC_cell_num
			if SC_cell_num == cell_num:
				HS_group, lib_grp, plate_num, well_num, cell_R, cell_area, cell_peri, cell_E = single_cell_stats[start_search][6:14]
				MASTER_SHEET.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(cell_UUID,
																														mito_UUID,
																														cell_filename,
																														mito_filename,
																														HS_group,
																														lib_grp,
																														plate_num,
																														well_num,
																														cell_num,
																														cell_R,
																														cell_area,
																														cell_peri,
																														cell_E,
																														mito_num,
																														volume,
																														nTriangles,
																														surface_area,
																														filepath))
				break
			else:
				pass

	MASTER_SHEET.close()


if __name__ == "__main__":
	main("L:\\Users\\gordon\\00000004 - Running Projects\\20180126 Mito quantification for Gordon\\20180306_results")
