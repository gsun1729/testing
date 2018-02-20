from render import *
from processing import *

def verify_shape(img_2d, stack_3d):
	z3, x3, y3 = stack_3d.shape
	x2, y2 = img_2d.shape
	if x2 == x3 and y2 == y3:
		return True
	else:
		return False


def stack_multiplier(image, stack):
	z, x, y = stack.shape
	composite = np.zeros_like(stack)
	if verify_shape(image, stack):
		for layer in xrange(z):
			composite[layer, :, :] = stack[layer, :, :] * image
	return composite
