from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import binary_blobs
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

def montage_x(tuple_img):
	num_img = len(tuple_img);
	for x in xrange(len(tuple_img)):
		plt.subplot(1, num_img, x + 1)
		plt.imshow(tuple_img[x])

	# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
	# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	# plt.colorbar(cax=cax)

	plt.show()

if __name__ == "__main__":
	test_image = binary_blobs(length = 200,
								blob_size_fraction = 0.1,
								n_dim = 3,
								volume_fraction = 0.3,
								seed = 1)
	stack_viewer(test_image)
