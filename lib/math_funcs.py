import numpy as np

def global_max(img_2d):
    return np.amax(img_2d.flatten())

def global_min(img_2d):
    return np.amin(img_2d.flatten())
