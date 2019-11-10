import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
from scipy.ndimage.interpolation import shift


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
    return image / 255  # here I divide by 255


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


# check what they saud on the whatsapp
# fill documentation
def multiply_matrices(conversion_matrix, image):
    height = image.shape[0]
    width = image.shape[1]
    pixel_stack = np.transpose(np.reshape(image, (height * width, 3)))
    result_pixel_stack = np.matmul(conversion_matrix, pixel_stack)
    result_pixel_stack = np.transpose(result_pixel_stack)
    return np.reshape(result_pixel_stack, (height, width, 3))


# I need to check that these functions are correct.
# do refactor here
def rgb2yiq(imRGB):
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
    return multiply_matrices(conversion_matrix, imRGB)


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
    return multiply_matrices(conversion_matrix, imYIQ)


# Histogram equalization part:

def histogram_eq_one_dimension(image, hist_orig):
    """
    A function that perform a histogram
    equalization on (Y axis of RGB image) or (gray scale)
    image.
    :param hist_orig The histogram of the given image.
    :param image: the given image as a
    2 dimensional matrix.
    :return: the matrix after the histogram equalization.
    """
    cumulative_histogram = np.cumsum(hist_orig)
    index_of_first_non_zero = np.argmin(cumulative_histogram)
    cumulative_histogram = shift(cumulative_histogram,
                                 -index_of_first_non_zero)
    mapping_table = cumulative_histogram / image.size
    mapping_table *= 255
    mapping_table = np.round(mapping_table).astype(int)
    return mapping_table[image]


def histogram_equalize_RGB(im_orig):
    """
    A function that do a histogram equalization
    on RGB image (with pixels in the range [0, 255].
    :param im_orig: Matrix that represents the image.
    :return: [im_eq, hist_orig, hist_eq] (as mentioned in
    the exercise).
    """
    im_orig_yiq = np.round(rgb2yiq(im_orig)).astype(int)
    hist_orig = np.histogram(im_orig_yiq[:, :, 0], bins=256, range=(0, 256))[0]
    im_eq_y = histogram_eq_one_dimension(im_orig_yiq[:, :, 0], hist_orig)
    hist_eq = np.histogram(im_eq_y, bins=256, range=(0, 256))[0]
    im_orig_yiq[:, :, 0] = im_eq_y
    im_eq = yiq2rgb(im_orig_yiq)
    return [im_eq, hist_orig, hist_eq]


def histogram_equalize_gray_scale(im_orig):
    """
    A function that do a histogram equalization
    on gray_scale image (with pixels in the range [0, 255].
    :param im_orig: Matrix that represents the image.
    :return: [im_eq, hist_orig, hist_eq] (as mentioned in
    the exercise).
    """
    hist_orig = np.histogram(im_orig, bins=256, range=(0, 256))[0]
    im_eq = histogram_eq_one_dimension(im_orig, hist_orig)
    hist_eq = np.histogram(im_eq, bins=256, range=(0, 256))[0]
    return [im_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):  # split this function to 2
    # functions: one for RGB and one for gray scale.
    """
    A function that do a histogram equalization.
    :param im_orig: Matrix that represents the given image.
    :return: [im_eq, hist_orig, hist_eq] (as mentioned in
    the exercise).
    """
    if type(im_orig) != np.ndarray:
        im_orig = np.array(im_orig)
    im_orig = np.round(im_orig * 255).astype(int)  # check the rounding issue.
    if im_orig.ndim == 3:
        result = histogram_equalize_RGB(im_orig)
    else:
        result = histogram_equalize_gray_scale(im_orig)
    result[0] /= 255
    result[0] = result[0].astype(np.float64)
    # result[0] = check_boundaries(result[0])  # implement a function that
    # checks that all the values are in the range: [0,1]
    return result


def quantize_one_dimension(im_orig, n_quant, n_iter):
    
    return [np.array([1]), [1]]


# Quantization part:
def quantize_RGB(im_orig, n_quant, n_iter):
    im_orig_yiq = np.round(rgb2yiq(im_orig)).astype(int)
    result = quantize_one_dimension(im_orig_yiq[:, :, 0], n_quant, n_iter)
    im_orig_yiq[:, :, 0] = result[0]
    im_quant = yiq2rgb(im_orig_yiq)
    result[0] = im_quant
    return result


def quantize_gray_scale(im_orig, n_quant, n_iter):
    return quantize_one_dimension(im_orig, n_quant, n_iter)


def quantize(im_orig, n_quant, n_iter):  # split this function to 2
    # functions: one for RGB and one for gray scale.
    if type(im_orig) != np.ndarray:
        im_orig = np.array(im_orig)
    im_orig = np.round(im_orig * 255).astype(int)  # check the rounding issue.
    if im_orig.ndim == 3:
        result = quantize_RGB(im_orig, n_quant, n_iter)
    else:
        result = quantize_gray_scale(im_orig, n_quant, n_iter)
    result[0] /= 255
    result[0] = result[0].astype(np.float64)
    return result


# tests:
Jerusalem = 'jerusalem.jpg'
monkey = 'monkey.jpg'
movie = 'low_contrast.jpg'

img = misc.imread(
    'C:\\Users\\USER\\Desktop\\Image_Processing\\ex1\\ex1_presubmit'
    '\\presubmit_externals\\' + Jerusalem)
img = histogram_equalize(img / 255)[0]
print(img)
plt.imshow(img)
plt.show()
