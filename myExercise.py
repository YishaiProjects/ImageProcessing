import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
from scipy.ndimage.interpolation import shift

img = misc.imread('my_image.jpg')

print(img)
print(img.dtype)
print(img.shape)
print(img.size)


#
# img = color.rgb2gray(img)
#
# print(img)
# print(img.dtype)
# print(img.shape)
# print(img.size)
#

# I need to check in every function that the input is a numpy array


def normalize(image):
    """
    A function that normalize the elements inside
    a numpy matrix, and change their type to float64.
    :param image: a numpy matrix
    :return: the matrix with all elements normalized
    to the range: [0, 1].
    """
    image = image.astype('float64')
    return image / 255  # here i divide by 255


def read_image(filename, representation):
    """
    A function that reads a file according to a
    given representation.
    :param filename: The file name.
    :param representation: 1 represents rgb,
    2 represents gray_scale.
    :return: An image according to the
    given representation.
    """
    image = misc.imread(filename)
    if representation == 2 or image.ndim != 3:  # here im checking
        # if im in rgb or gray with
        # the dimensions only
        if np.any(image > 1):
            return normalize(image)
    elif representation == 1:
        return color.rgb2gray(image)


def imdisplay(filename, representation):
    """
    A function that displays an image from a
    given file, according to the given representation.
    :param filename: The name of the given file.
    :param representation: The desired representation,
    1 for gray, 2 for rgb.
    :return: Nothing.
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap='gray')
    plt.show()


def mult_matrices(conversion_matrix, image):
    print("image: ", image)
    height = image.shape[0]
    width = image.shape[1]
    pixel_stack = np.transpose(np.reshape(image, (height * width, 3)))
    print("image after trans: ", image)
    result_pixel_stack = np.matmul(conversion_matrix, pixel_stack)
    print("image after mul: ", image)
    result_pixel_stack = np.transpose(result_pixel_stack)
    print("image after another trans: ", image)
    print("image at the end: ",
          np.reshape(result_pixel_stack, (height, width, 3)))
    return np.reshape(result_pixel_stack, (height, width, 3))


# I need to check that these functions are correct.
# do refactor here
def rgb2yiq1(imRGB):
    """
    A function that converts an RGB matrix
    to YIQ matrix. (The RGB needs to be in the
    range: [0, 1] in order to se  the image good)
    :param imRGB: The given RGB matrix.
    :return: The YIQ matrix.
    """
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])
    return mult_matrices(conversion_matrix, imRGB)  #


# (This is another implementation of rgb2yiq)
def rgb2yiq2(imRGB):
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])
    height = imRGB.shape[0]
    width = imRGB.shape[1]
    pixel_stack = np.reshape(imRGB, (height * width, 3))
    l = []  # maybe fix here the list to
    # something else
    for i in range(3):
        Rs = pixel_stack[:, 0] * conversion_matrix[i, 0]
        Gs = pixel_stack[:, 0] * conversion_matrix[i, 1]
        Bs = pixel_stack[:, 0] * conversion_matrix[i, 2]
        l.append(np.add(np.add(Rs, Gs), Bs))
    result = np.vstack(l)
    result = np.transpose(result)
    return np.reshape(result, (height, width, 3))


def yiq2rgb(imYIQ):
    """
    A function that converts a YIQ image
    to RGB.
    :param imYIQ: the given image matrix.
    :return: The RGB representation of the image.
    """
    conversion_matrix = np.array([[1, 0.956, 0.619],
                                  [1, -0.272, -0.647],
                                  [1, -1.106, 1.703]])
    return mult_matrices(conversion_matrix, imYIQ)


def histogram_eq_helper(image, hist_orig):
    """
    A function that perform a histogram
    equalization on Y axis of RGB image \ gray scale
    image.
    :param hist_orig The histogram of the given image.
    :param image: the given image as a
    2 dimensional matrix.
    :return: the matrix after the histogram equalization.
    """
    cumulative_histogram = np.cumsum(hist_orig[0])
    index_of_first_non_zero = np.argmin(cumulative_histogram)
    cumulative_histogram = shift(cumulative_histogram,
                                 -index_of_first_non_zero)
    mapping_table = cumulative_histogram / image.size
    mapping_table *= 255
    mapping_table = np.round(mapping_table)
    return mapping_table[image]


def histogram_equalize(im_orig):            # split this function to 2
    # functions: one for RGB and one for gray scale.
    if type(im_orig) != np.ndarray:
        im_orig = np.array(im_orig)
    im_orig = im_orig * 255  # check that 255 is good
    # and not 256
    if im_orig.ndim == 3:
        im_orig = rgb2yiq1(im_orig)
        hist_orig = np.histogram(im_orig, bins=256, range=(0, 256))
        im_eq_y = histogram_eq_helper(im_orig[:, :, 0], hist_orig)
        hist_eq = np.histogram(im_eq_y, bins=256, range=(0, 256))
        im_orig[:, :, 0] = im_eq_y
        im_eq = yiq2rgb(im_orig)
    else:
        hist_orig = np.histogram(im_orig, bins=256, range=(0, 256))
        im_eq = histogram_eq_helper(im_orig, hist_orig)
        hist_eq = np.histogram(im_eq, bins=256, range=(0, 256))
    im_eq /= 255
    im_eq = check_boundries(im_eq)        #implement a function that checks
    #that all the values are in the range: [0,1]
    return [im_eq, hist_orig, hist_eq]

# img = yiq2rgb(rgb2yiq1(read_image('my_image.jpg', 2)))
# plt.imshow(img, cmap='gray')
# plt.show()
