import numpy as np
import sol4
import sol4_utils

import sol4_n
import sol4_utils_n

import matplotlib.pyplot as plt

# im1 = sol4_utils.read_image("external\\oxford1.jpg", 1)
# im2 = sol4_utils.read_image("external\\oxford2.jpg", 1)
#
# pyr1 = sol4_utils_n.build_gaussian_pyramid(im1, 3, 3)[0]
# pyr2 = sol4_utils_n.build_gaussian_pyramid(im2, 3, 3)[0]
#
# points1, descriptors1 = sol4.find_features(pyr1)
# points2, descriptors2 = sol4.find_features(pyr2)
#
# X1 = points1[:, 0]
# Y1 = points1[:, 1]
#
# X2 = points2[:, 0]
# Y2 = points2[:, 1]
#
# indexes1, indexes2 = sol4.match_features(descriptors1, descriptors2, 0.5)
#
#
# h, inliers = sol4.ransac_homography(points1[indexes1], points2[indexes2], 50, 10, translation_only=False)
#
# sol4.display_matches(im1, im2, points1[indexes1], points2[indexes2], inliers)


x, y = np.meshgrid(np.array([1,2,3,4,5,6]), np.array([1,2,3,4,5,6, 7]))
n = x.reshape(-1)
warp_cord = np.zeros((2, x.reshape(-1).shape[0]))
print(n)
