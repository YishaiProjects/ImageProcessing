import scipy.misc as misc
import numpy as np
from scipy.ndimage.interpolation import shift

img1 = misc.imread('my_image.jpg')

img = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 1, 12],
                [13, 14, 15, 16]])
im = np.array([])

print(im)
