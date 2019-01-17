import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def stack_viewer_2x(image, image2):
	'''
	Module to allow for scrolling through two 3d stack images
	Modified from the following source:
	https://matplotlib.org/gallery/animation/image_slices_viewer.html

	:param image1: [np.ndarray] 3d stack image for viewing
	:param image2: [np.ndarray] 3d stack image for viewing
	'''
	try:
		z1, x1, y1 = image.shape
		z2, x2, y2 = image2.shape
		print(image.shape)
		print(image2.shape)
	except ValueError:
		print("Improper dimensions, non-stack 3D image")
		print(image.shape)
		print(image2.shape)
		raise Exception

	class IndexTracker(object):
		def __init__(self, ax1, ax2, img_stack1, img_stack2):
			self.ax1 = ax1
			self.ax2 = ax2
			ax1.set_title('scroll to navigate images')
			ax2.set_title('scroll to navigate images')

			# self.axes = axes
			# axes.set_title('scroll to navigate images')
			#
			self.img_stack1 = img_stack1
			self.img_stack2 = img_stack2
			self.slices1, rows1, cols1 = img_stack1.shape
			self.slices2, rows2, cols2 = img_stack2.shape

			self.start_index = self.slices1//2
			#
			self.im1 = ax1.imshow(self.img_stack1[self.start_index,:, :])
			self.im2 = ax2.imshow(self.img_stack2[self.start_index,:, :])
			self.update()

		def onscroll(self, event):
			# print("%s %s" % (	event.button, event.step))
			if event.button == 'up':
				self.start_index = (self.start_index + 1) % self.slices1
			else:
				self.start_index = (self.start_index - 1) % self.slices2
			self.update()

		def update(self):
			self.im1.set_data(self.img_stack1[ self.start_index,:, :])
			self.im2.set_data(self.img_stack2[ self.start_index,:, :])
			ax1.set_ylabel('slice %s' % self.start_index)
			ax2.set_ylabel('slice %s' % self.start_index)
			self.im1.axes.figure.canvas.draw()
			self.im2.axes.figure.canvas.draw()

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
	tracker = IndexTracker(ax1, ax2, image, image2)
	fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
	plt.show()


def make_ticklabels_invisible(fig):
	'''Helper function for montage_n_x, removes tick labels
	https://matplotlib.org/users/gridspec.html

	:param fig: [matplotlib.fig] figure to have tick labels removed
	'''
	for i, ax in enumerate(fig.axes):
		ax.text(0.5, 0.5, "ax%d" % (i + 1), va = "center", ha = "center")
		for tl in ax.get_xticklabels() + ax.get_yticklabels():
			tl.set_visible(False)


def main():
	# data = io.imread("/home/gsun/Documents/Github/mitochondria-image-processing/data/_hs/P19F8_1_w2561 Laser.TIF")
	# data2 = io.imread("/home/gsun/Documents/Github/mitochondria-image-processing/data/_hs/P19F8_1_w2561 Laser.TIF")
	#
	# stack_viewer_2x(data, data2)
	os.system("cp test_file.txt ./misc/test_file_moved.txt")

	pairing_LUT = open('/home/gsun/Desktop/20181013 Rerun MD 13 BACKUP/MASTER_RESULTS.txt', 'r')
	namepair_data = pairing_LUT.readlines()
	pairing_LUT.close()
	data_table = [x.strip('\n').split('\t') for x in namepair_data]

	mito_IDs = {}
	for indx, row in enumerate(data_table):
		try:
			ID = row[1]
		except IndexError:
			print(row)
			raise Exception

		if ID in mito_IDs:
			pass
		else:
			# if the ID of the mitochondria was found, rename the directory accounting for the new location and create a dictionary with the mito key number
			filename = row[3]
			location = row[-1].replace('\\', '/')
			complete_path = "/home/gsun/Desktop" + location[21:] + "/" + filename
			mito_IDs[ID] = complete_path
	for key, value in mito_IDs.items():
		try:
			new_filename = "M_" + key + "_RAW" + ".TIF"
			new_fileloc = "/home/gsun/Desktop/20181013 Rerun MD 13 BACKUP/mito/raw_imgs/" + new_filename
			value_spaces = value.replace(' ', '\\ ')
			new_fileloc_spaces = new_fileloc.replace(' ', '\\ ')
			print("cp {} > {}".format(value_spaces, new_fileloc_spaces))
			os.system("cp {} {}".format(value_spaces, new_fileloc_spaces))

			# raise Exception
		except IOError:
			print("Could not read file")
			print(value)
			raise Exception
		# stack_viewer_2x(image, image)
	print(len(mito_IDs))
		# raise Exception

	# print(data_table[0][-1].replace('\\', '/'))



if __name__ == "__main__":
	main()
