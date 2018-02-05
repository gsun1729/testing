from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
'''
Module to allow for scrolling through a 3d stack image
https://matplotlib.org/gallery/animation/image_slices_viewer.html
'''
def main(image):
    class IndexTracker(object):
        def __init__(self, axes, image_stack):
            self.axes = axes
            axes.set_title('use scroll wheel to navigate images')

            self.image_stack = image_stack
            rows, cols, self.slices = image_stack.shape
            self.start_index = self.slices//2

            self.im = axes.imshow(self.image_stack[:, :, self.start_index])
            self.update()

        def onscroll(self, event):
            print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.start_index = (self.start_index + 1) % self.slices
            else:
                self.start_index = (self.start_index - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.image_stack[:, :, self.start_index])
            axes.set_ylabel('slice %s' % self.start_index)
            self.im.axes.figure.canvas.draw()


    fig, axes = plt.subplots(1, 1)
    tracker = IndexTracker(axes, image)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

if __name__ == "__main__":
    main()
