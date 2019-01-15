import numpy as np


def stack_viewer(image1, image2):
	'''
	Module to allow for scrolling through two 3d stack images
    Modified from the following source:
	https://matplotlib.org/gallery/animation/image_slices_viewer.html

	:param image1: [np.ndarray] 3d stack image for viewing
    :param image2: [np.ndarray] 3d stack image for viewing
	'''
	try:
		z1, x1, y1 = image1.shape
        z2, x2, y2 = image2.shape
	except ValueError:
		print("Improper dimensions, non-stack 3D image")
		print(image1.shape)
        print(image2.shape)
		raise Exception

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
			# print("%s %s" % (	event.button, event.step))
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
