from lib.render import *
import scipy.io

def main():
	mito_binary = scipy.io.loadmat("C:\\Users\\Gordon Sun\\Documents\\GitHub\\bootlegged_pipeline\\ooga\\mito\\M_336b6da548994c08acd5912b7f6ae2a0_bin.mat")['data']
	mito_skel = scipy.io.loadmat("C:\\Users\\Gordon Sun\\Documents\\GitHub\\bootlegged_pipeline\\ooga\\mito\\M_336b6da548994c08acd5912b7f6ae2a0_skel.mat")['data']
	cell_binary = scipy.io.loadmat("C:\\Users\\Gordon Sun\\Documents\\GitHub\\bootlegged_pipeline\\ooga\\cell\\C_7b45e26a204f40dca88aa19ce7b93463_dat.mat")['data']
	view_2d_img(cell_binary)
if __name__ == "__main__":
	main()
