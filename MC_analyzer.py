import sys
sys.path.insert(0, '.\\lib')
from render import *
from read_write import *
from math_funcs import *
from UUID import *
import scipy.io
# from analysis_f import *
from processing import *
from skimage.morphology import skeletonize_3d

def main():
	options = sys.argv
	if len(options) != 3:
		print "> Wrong # of arguments, please provide input in the form \n> python main.py read_directory write_directory\n"
		for num, item in enumerate(options):
			print "> \tArg{}: {}\n".format(num, item)
		sys.exit()
	else:
		UID_file_loc = options[-2]
		save_dir = options[-1]

	root_read_dir = os.path.dirname(UID_file_loc)
	cell_data_dir = os.path.join(root_read_dir, "cell")
	mito_data_dir = os.path.join(root_read_dir, "mito")

	cell_filelist = get_img_filenames(cell_data_dir)
	mito_filelist = get_img_filenames(mito_data_dir)
	UUID_datatable = read_UUID_file(UID_file_loc)

	C_M_UUID_pairs = create_pairTable(cell_filelist, UUID_datatable)
	UUID_pairs = create_densepairTable(cell_filelist, UUID_datatable)

	filename_pairs = []
	for cell_UUID, mito_UUID in UUID_pairs:
		filename_pairs.append(["C_" + cell_UUID + "_dat.mat",
								"M_" + mito_UUID + "_bin.mat",
								"M_" + mito_UUID + "_skel.mat"])

	write_list_txt(save_dir, "Cell_mito_UUID_Pairs.txt", C_M_UUID_pairs)
	write_list_txt(save_dir, "UUID_paired_filenames.txt", filename_pairs)

	for filenames in filename_pairs:
		save_fileID = get_UUID(filenames[0])
		cell_img = scipy.io.loadmat(os.path.join(cell_data_dir, filenames[0]))['data']
		mito_stack = scipy.io.loadmat(os.path.join(mito_data_dir, filenames[1]))['data']
		mito_skel = scipy.io.loadmat(os.path.join(mito_data_dir, filenames[2]))['data']

		labeled_mito_bin = stack_multiplier(cell_img, mito_stack)
		labeled_mito_skel = stack_multiplier(cell_img, mito_skel)

		save_data(labeled_mito_bin, "CM_" + save_fileID + "_bin", save_dir)
		save_data(labeled_mito_skel, "CM_" + save_fileID + "_skel", save_dir)

if __name__ == "__main__":
	main()
