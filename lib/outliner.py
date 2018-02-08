import numpy as np
import sys
def max_projection(image, axis = 0):
    '''
    Axis is defined as the index of the image.shape output.
    By default it is the Z axis (z,x,y)
    '''
    try:
        return np.sum(image,axis)
    except ValueError:
        if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
            print "Axis value invalid"
        else:
            print "Image input faulty"
        sys.exit()

def avg_projection(image, axis = 0):
    '''
    Axis is defined as the index of the image.shape output.
    By default it is the Z axis (z,x,y)
    '''
    try:
        print axis
        z, x, y = image.shape
        return np.sum(image,axis)//z
    except ValueError:
        if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
            print "Axis value invalid"
        else:
            print "Image input faulty"
        sys.exit()
