import scipy.misc as misc
import numpy as np
from scipy.ndimage.interpolation import shift


im2 = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15], [16,17,18]]])
img = np.array([[3, 3, 3, 2],
                [2, 3, 3, 2],
                [9, 10, 1, 12],
                [13, 9, 10, 10]])
print(im2[:,:,0])
