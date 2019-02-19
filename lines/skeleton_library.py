from scipy import io
from lib.read_write import *
from lib.render import *
from lib.processing import *
# import lines.mito_counter
from lib.pathfinder import *
import pygraphviz as pgv


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



filename = get_img_filenames("/home/gsun/Desktop/20190210_Yeast_Phase_Mito_imgs/20190210_Yeast_Phase_MTSmCherry_analysis", suffix = '_skel.mat')

for filename_info in filename:
	print(filename_info)
	temp = filename_info[-1].replace('_skel.mat','_bin.mat')
	print(temp)
	binary = scipy.io.loadmat(temp)['data']
	skeleton = scipy.io.loadmat(filename_info[-1])['data']
	# view_2d_img(max_projection(skeleton))
	# raise Exception
	for skel_indx in np.unique(skeleton):
		if skel_indx == 0:
			pass 
		else:

			temp = get_bounding_img(skeleton, skel_indx)
			v, graph = imglattice2graph(temp, neighbor_distance = ['1U', '3U'])
			process_graph(graph)
			# input()
			print('\n')			
			# raise Exception

		
		
		# print(i)
		# raise Exception	
	# print(temp.shape)
	# print(skeleton.shape)
	# stack_viewer(temp)
	# stack_viewer(skeleton)
	
	raise Exception