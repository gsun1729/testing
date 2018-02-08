import numpy as np
import sys
from skimage.exposure import adjust_gamma
from skimage import io
dtype2bits = {'uint8': 8,
              'uint16': 16,
              'uint32': 32}

def gamma_stabilize(image, alpha_clean=5, floor_method='min'):
    """
    Normalizes the luma curve. floor intensity becomes 0 and max allowed by the bit number - 1

    :param image:
    :param alpha_clean: size of features that would be removed if surrounded by a majority of
    :param floor_method: ['min', '1q', '5p', 'median'] method of setting the floor intensity. 1q is first quartile, 1p is the first percentile
    :return:
    """
    bits = dtype2bits[image.dtype.name]
    if floor_method == 'min':
        inner_min = np.min(image)
    elif floor_method == '1q':
        inner_min = np.percentile(image, 25)
    elif floor_method == '5p':
        inner_min = np.percentile(image, 5)
    elif floor_method == 'median':
        inner_min = np.median(image)
    else:
        raise PipeArgError('floor_method can only be one of the three types: min, 1q, 5p or median')
    stabilized = (image - inner_min) / (float(2 ** bits) - inner_min)
    stabilized[stabilized < alpha_clean*np.median(stabilized)] = 0
    return stabilized


def sum_projection(image, axis = 0):
    '''
    Axis is defined as the index of the image.shape output.
    By default it is the Z axis (z,x,y)
    '''
    try:
        return np.sum(image, axis)
    except ValueError:
        if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
            print "Axis value invalid"
        else:
            print "Image input faulty"
        sys.exit()


def max_projection(image, axis = 0):
    '''
    Axis is defined as the index of the image.shape output.
    By default it is the Z axis (z,x,y)
    '''
    try:
        return np.amax(image, axis)
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
        return np.sum(image, axis)//z
    except ValueError:
        if not((axis >= 0 or axis <= 2) and isinstance(axis, int)):
            print "Axis value invalid"
        else:
            print "Image input faulty"
        sys.exit()

def process_cell_outline(img_filepath):
    cell = io.imread(img_filepath)
