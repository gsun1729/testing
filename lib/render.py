from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import binary_blobs
from matplotlib.gridspec import GridSpec
import sys
'''
Module to allow for scrolling through a 3d stack image modified from the following source:
https://matplotlib.org/gallery/animation/image_slices_viewer.html
'''
def stack_viewer(image):
	try:
		z,x,y = image.shape
	except ValueError:
		print("Improper dimensions, non-stack Image")
		print(image.shape)
		sys.exit()

	class IndexTracker(object):
		def __init__(self, axes, image_stack):
			self.axes = axes
			axes.set_title('scroll to navigate images')

			self.image_stack = image_stack
			self.slices, rows, cols = image_stack.shape
			self.start_index = self.slices//2

			self.im = axes.imshow(self.image_stack[self.start_index,:, :])
			self.update()

		def onscroll(self, event):
			print("%s %s" % (event.button, event.step))
			if event.button == 'up':
				self.start_index = (self.start_index + 1) % self.slices
			else:
				self.start_index = (self.start_index - 1) % self.slices
			self.update()

		def update(self):
			self.im.set_data(self.image_stack[ self.start_index,:, :])
			axes.set_ylabel('slice %s' % self.start_index)
			self.im.axes.figure.canvas.draw()

	fig, axes = plt.subplots(1, 1)
	tracker = IndexTracker(axes, image)
	fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
	plt.show()


def view_2d_img(img):
	imgplot = plt.imshow(img)
	plt.show()


def make_ticklabels_invisible(fig):
	'''
	Helper function for montage_n_x
	https://matplotlib.org/users/gridspec.html
	'''
	for i, ax in enumerate(fig.axes):
		ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
		for tl in ax.get_xticklabels() + ax.get_yticklabels():
			tl.set_visible(False)


def montage_n_x(*tuple_img_line):
	num_rows = len(tuple_img_line)
	num_cols = 0;
	for lines in tuple_img_line:
		if len(lines) > num_cols:
			num_cols = len(lines)
	# plt.figure()
	grid = GridSpec(num_rows, num_cols)
	for row in xrange(num_rows):
		for col in xrange(num_cols):
			try:
				plt.subplot(grid[row,col])

				plt.imshow(tuple_img_line[row][col])
			except IndexError:
				print("Exceed index")
				break
	make_ticklabels_invisible(plt.gcf())

	plt.show()


if __name__ == "__main__":
	test_image = binary_blobs(length = 200,
								blob_size_fraction = 0.1,
								n_dim = 3,
								volume_fraction = 0.3,
								seed = 1)
	stack_viewer(test_image)
