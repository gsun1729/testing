from scipy import io
from skimage import io
from lib.read_write import *
from lib.render import *
from lib.processing import *
# import lines.mito_counter
from lib.pathfinder import *
import pygraphviz as pgv
from tools.scroll_compare.render_functions import *


def main():
	filename = get_img_filenames("/home/gsun/Desktop/20181013_Rerun_MD_13_BACKUP/mito/skeletons/", suffix = '_skel.mat')
	labeled_skeleton_savedir = "/home/gsun/Desktop/20181013_Rerun_MD_13_BACKUP/mito/labeled_skeletons"
	labeled_binary_savedir = "/home/gsun/Desktop/20181013_Rerun_MD_13_BACKUP/mito/labeled_binary_imgs"

	for UUID, UUID_prefix, _, _, _,file_path in filename:
		print(UUID_prefix)
		skel_path = file_path
		binary_path = file_path.replace("skeletons", "binary_imgs").replace("_skel.mat","_bin.mat")
		raw_path = file_path.replace("skeletons", "raw_imgs").replace("_skel.mat", "_RAW.TIF")
		binary = scipy.io.loadmat(binary_path)['data']
		skeleton = scipy.io.loadmat(skel_path)['data']
		raw = io.imread(raw_path)
		
		raw_seg = stack_stack_multply(raw, binary)
		# stack_viewer_2x(raw_seg, raw)
		# raise

		labeled_binary = layer_comparator(binary)
		labeled_skeleton = stack_stack_multply(labeled_binary, skeleton)

		# save_data(labeled_binary, UUID_prefix + "_lbin.mat", labeled_binary_savedir)
		# save_data(labeled_skeleton, UUID_prefix + "_lskel.mat", labeled_skeleton_savedir)
		
		for skel_indx in np.unique(labeled_skeleton):
			print("\tSkeleton {}".format(skel_indx))
			if skel_indx == 0:
				pass 
			else:
				one_skel = get_bounding_img(labeled_skeleton, skel_indx)
				stack_viewer(one_skel)
				stack_viewer(get_bounding_img(labeled_binary, skel_indx))
				v, graph = imglattice2graph(one_skel, neighbor_distance = ['1U', '3U'])
				process_graph(graph)
				# # input()
				# print('\n')			
				# raise Exception

			
			
			# print(i)
			# raise Exception	
		# print(temp.shape)
		# print(skeleton.shape)
		# stack_viewer(temp)
		# stack_viewer(skeleton)
		
		raise Exception


def test():
	temp = np.zeros([6,6,6])
	temp[0,0,0] = 1
	temp[1,1,1] = 1
	temp[2,2,2] = 1
	temp[3,3,3] = 1
	temp[4,4,4] = 1
	temp[5,5,3] = 1
	temp[5,3,5] = 1
	temp[3,5,5] = 1

	temp2 = np.zeros([6,6,6])
	temp2[0,0,0] = 1
	temp2[1,0,0] = 1
	temp2[2,0,0] = 1
	temp2[3,0,0] = 1
	temp2[4,0,0] = 1
	temp2[5,0,0] = 1
	temp2[4,1,0] = 1
	temp2[4,0,1] = 1


	temp3 = np.zeros([5,5,5])
	for i in range(5):
		temp3[i,2,2] = 1

	temp3[1,2,0] = 1
	temp3[1,4,2] = 1

	temp3[2,2,1] = 1
	temp3[2,3,2] = 1

	temp3[3,2,0] = 1
	temp3[3,4,2] = 1

	# def temp_test(temp):
	# 	# stack_viewer(temp)
	v, graph = imglattice2graph(temp3, neighbor_distance = ['1U', '3U'])
	# 	# print(dict(graph.get_self()))
	# 	graph = prune_graph(graph)
	# 	print(graph.get_self())
	# 	print(len(graph.num_junctions()))
	# 	print(len(graph.num_endpoints()))

	# 		# pass
	# 	ddict2nx_graph(graph.get_self())
		
		
		# ddict2nx_graph(graph.get_self())
	# print(type(graph))
	# a = process_graph(graph)
	# # print(a)
	# raise 



if __name__ == "__main__":
	main()