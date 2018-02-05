import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.data import binary_blobs
import image_scroll


def main(image):


# sys.exit()
skeleton3d = skeletonize_3d(image)
