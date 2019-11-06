import scipy.misc as misc
import numpy as np
from scipy.ndimage.interpolation import shift

img1 = misc.imread('my_image.jpg')

im2 = np.array([[[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
img = np.array([[3, 3, 3, 2],
                [2, 3, 3, 2],
                [9, 10, 1, 12],
                [13, 9, 10, 10]])
map_table = np.array([1, 1, 1, 2,2,2,2,2,3,3,3,3,3, 3])
print(im2*2)

